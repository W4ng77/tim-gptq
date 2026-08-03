"""Shared low-bit quantization and integer-bundle utilities."""

import math

import torch

from modelutils import find_layers
from rotation import largest_power_of_two_divisor


def get_encoder_input(batch):
    if "input_features" in batch:
        return batch["input_features"]
    if "input_values" in batch:
        return batch["input_values"]
    raise KeyError("Calibration batch missing input_features/input_values.")


def _squeeze_batch(x):
    if x.dim() == 3 and x.size(0) == 1:
        return x[0]
    return x


def _to_cpu_detached(v):
    if torch.is_tensor(v):
        return v.detach().cpu()
    if isinstance(v, tuple):
        return tuple(_to_cpu_detached(x) for x in v)
    if isinstance(v, list):
        return [_to_cpu_detached(x) for x in v]
    if isinstance(v, dict):
        return {k: _to_cpu_detached(x) for k, x in v.items()}
    return v


def _move_to_device(v, dev, dtype):
    if torch.is_tensor(v):
        return v.to(dev)
    if isinstance(v, tuple):
        return tuple(_move_to_device(x, dev, dtype) for x in v)
    if isinstance(v, list):
        return [_move_to_device(x, dev, dtype) for x in v]
    if isinstance(v, dict):
        return {k: _move_to_device(x, dev, dtype) for k, x in v.items()}
    return v


def _move_layer_kwargs(layer_kwargs, dev, dtype):
    moved = {}
    for k, v in layer_kwargs.items():
        moved[k] = _move_to_device(v, dev, dtype)
    return moved


def pseudo_quantize_tensor(
    w, n_bit=8, zero_point=True, q_group_size=-1, inplace=False, get_scale_zp=False
):
    if w.dim() != 2:
        raise ValueError(f"Expected a 2D weight matrix, got shape {tuple(w.shape)}")

    group_size = w.shape[1] if q_group_size <= 0 else int(q_group_size)
    quantized_groups = []
    scale_groups = []
    zero_groups = []

    for start in range(0, w.shape[1], group_size):
        group = w[:, start : start + group_size]
        if zero_point:
            max_value = group.amax(dim=1, keepdim=True)
            min_value = group.amin(dim=1, keepdim=True)
            max_int = 2**n_bit - 1
            min_int = 0
            scales = (max_value - min_value).clamp(min=1e-5) / max_int
            zeros = (-torch.round(min_value / scales)).clamp(min_int, max_int)
        else:
            max_value = group.abs().amax(dim=1, keepdim=True).clamp(min=1e-5)
            max_int = 2 ** (n_bit - 1) - 1
            min_int = -(2 ** (n_bit - 1))
            scales = max_value / max_int
            zeros = torch.zeros_like(scales)

        quantized = (
            torch.clamp(torch.round(group / scales) + zeros, min_int, max_int)
            - zeros
        ) * scales
        quantized_groups.append(quantized)
        scale_groups.append(scales)
        zero_groups.append(zeros)

    quantized_weight = torch.cat(quantized_groups, dim=1)
    scales = torch.cat(scale_groups, dim=1)
    zeros = torch.cat(zero_groups, dim=1)
    if inplace:
        w.copy_(quantized_weight)
        quantized_weight = w
    if get_scale_zp:
        return quantized_weight, scales, zeros
    return quantized_weight


def _candidate_layer_names(name):
    aliases = {
        "self_attn.q_proj": ["self_attn.q_proj", "self_attn.query"],
        "self_attn.k_proj": ["self_attn.k_proj", "self_attn.key"],
        "self_attn.v_proj": ["self_attn.v_proj", "self_attn.value"],
        "self_attn.out_proj": ["self_attn.out_proj", "self_attn.o_proj"],
        "encoder_attn.q_proj": ["encoder_attn.q_proj", "cross_attn.q_proj", "cross_attn.query"],
        "encoder_attn.k_proj": ["encoder_attn.k_proj", "cross_attn.k_proj", "cross_attn.key"],
        "encoder_attn.v_proj": ["encoder_attn.v_proj", "cross_attn.v_proj", "cross_attn.value"],
        "encoder_attn.out_proj": ["encoder_attn.out_proj", "encoder_attn.o_proj"],
        "fc1": ["fc1", "mlp.fc1"],
        "fc2": ["fc2", "mlp.fc2"],
    }
    return aliases.get(name, [name])


def _resolve_layer_group_if_present(full, names):
    resolved = []
    for name in names:
        found = None
        for cand in _candidate_layer_names(name):
            if cand in full:
                found = cand
                break
        if found is None:
            return None
        resolved.append(found)
    return resolved


def _is_fc2_name(name):
    return name == "fc2" or name.endswith(".fc2")


def _is_ffn_output_name(name):
    return _is_fc2_name(name) or name == "down_proj" or name.endswith(".down_proj")


def _should_protect_ffn_output(
    args,
    module_name,
    layer_idx=None,
    total_layers=None,
):
    if not _is_ffn_output_name(module_name):
        return False
    if bool(getattr(args, "ffn_output_fp16", False)):
        return True
    if not bool(getattr(args, "tail_ffn_output_fp16", False)):
        return False
    if layer_idx is None or total_layers is None:
        return False
    fraction = float(getattr(args, "tail_ffn_fraction", 0.25))
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"tail_ffn_fraction must be in (0, 1], got {fraction}")
    tail_layers = max(1, int(math.ceil(int(total_layers) * fraction)))
    return int(layer_idx) >= int(total_layers) - tail_layers


def _record_fp16_protected_module(args, module_name, module):
    metadata = getattr(args, "_quantization_metadata", {})
    protection = metadata.setdefault(
        "fp16_protection",
        {
            "policy": "ffn_output_projection",
            "modules": {},
        },
    )
    protection["modules"][module_name] = int(module.weight.numel())
    args._quantization_metadata = metadata


def _should_apply_qep(args, module_name):
    if not bool(getattr(args, "qep", False)):
        return False
    target = getattr(args, "qep_target", "legacy-except-fc2")
    if target == "legacy-except-fc2":
        return not _is_fc2_name(module_name)
    if target == "ffn-output":
        return _is_ffn_output_name(module_name)
    raise ValueError(f"Unsupported QEP target: {target!r}")


def _record_rotation_module(args, module_key, in_features):
    metadata = getattr(args, "_quantization_metadata", {})
    rotation = metadata.setdefault(
        "rotation",
        {
            "scheme": "randomized_block_diagonal_hadamard",
            "seed": int(getattr(args, "rotation_seed", 0)),
            "seed_derivation": "sha256(seed:module_key) per module",
            "block_size_policy": "largest_power_of_two_divisor_of_in_features",
            "fold_back": "W_hat = quantize(W R) R^T; inference path unchanged",
            "groupsize_alignment": (
                "quantization groups partition rotated columns; group "
                "boundaries need not align with Hadamard block boundaries"
            ),
            "modules": {},
        },
    )
    rotation["modules"][module_key] = {
        "in_features": int(in_features),
        "hadamard_block_size": int(largest_power_of_two_divisor(int(in_features))),
    }
    args._quantization_metadata = metadata


def _record_qep_gate_selection(args, module_name, selected_alpha, scores):
    metadata = getattr(args, "_quantization_metadata", {})
    gate = metadata.setdefault(
        "ffncomp_gate",
        {
            "criterion": "calibration_local_output_reconstruction_surrogate",
            "candidates": list(getattr(args, "qep_gate_candidates", ())),
            "selected_by_module": {},
        },
    )
    gate["selected_by_module"][module_name] = {
        "alpha": float(selected_alpha),
        "candidate_scores": {
            str(float(alpha)): float(score) for alpha, score in scores.items()
        },
    }
    args._quantization_metadata = metadata


def _build_module_key(scope, layer_idx, module_name):
    return f"{scope}.layer{int(layer_idx)}.{module_name}"


def _lookup_alpha(alpha_by_layer, default_alpha, scope, layer_idx, module_name):
    if alpha_by_layer is None:
        return float(default_alpha)

    module_key = _build_module_key(scope, layer_idx, module_name)
    if module_key in alpha_by_layer:
        return float(alpha_by_layer[module_key])

    legacy_layer_key = f"{scope}.{int(layer_idx)}"
    if legacy_layer_key in alpha_by_layer:
        return float(alpha_by_layer[legacy_layer_key])

    new_layer_key = f"{scope}.layer{int(layer_idx)}"
    if new_layer_key in alpha_by_layer:
        return float(alpha_by_layer[new_layer_key])

    return float(default_alpha)


@torch.no_grad()
def _collect_module_weight_norms(model):
    norms = {}

    encoder = model.model.encoder
    for i, layer in enumerate(encoder.layers):
        full = find_layers(layer)
        for name, mod in full.items():
            key = _build_module_key("enc", i, name)
            norms[key] = float(mod.weight.detach().float().norm().item())

    decoder = model.model.decoder
    for i, layer in enumerate(decoder.layers):
        full = find_layers(layer)
        for name, mod in full.items():
            key = _build_module_key("dec", i, name)
            norms[key] = float(mod.weight.detach().float().norm().item())

    return norms


def _normalize_positive_dict(vals, eps=1e-12):
    if not vals:
        return {}
    raw = list(vals.values())
    raw = [float(v) for v in raw]
    vmax = max(raw)
    vmin = min(raw)
    if vmax - vmin < eps:
        return {k: 1.0 for k in vals.keys()}
    out = {}
    for k, v in vals.items():
        z = (float(v) - vmin) / (vmax - vmin + eps)
        out[k] = 0.5 + z
    return out


def map_scores_to_alpha(
    scores,
    alpha_min=0.1,
    alpha_max=0.8,
    fallback_alpha=0.5,
    min_spread=1e-12,
    log_scale=True,
    invert=False,
):
    scores = torch.tensor(scores, dtype=torch.float32)

    if len(scores) == 0:
        return []
    if len(scores) == 1:
        return [float(fallback_alpha)]

    if log_scale:
        scores = torch.log(scores + 1e-12)

    smin = scores.min()
    smax = scores.max()
    spread = float((smax - smin).item())

    if spread < min_spread:
        return [float(fallback_alpha)] * len(scores)

    z = (scores - smin) / (smax - smin + 1e-12)

    if invert:
        alphas = alpha_max - (alpha_max - alpha_min) * z
    else:
        alphas = alpha_min + (alpha_max - alpha_min) * z

    return [float(a) for a in alphas]


def _dedupe_module_pairs(module_name_pairs):
    seen = set()
    unique_pairs = []
    for q_name, fp_name in module_name_pairs:
        pair = (q_name, fp_name)
        if pair in seen:
            continue
        seen.add(pair)
        unique_pairs.append(pair)
    return unique_pairs


def _register_module_mse_hooks(full_q, full_fp, unique_pairs):
    module_out_q = {}
    module_out_fp = {}

    def make_out_hook(store, name):
        def hook(_, __, out):
            if isinstance(out, tuple):
                out = out[0]
            store[name] = out.detach()

        return hook

    handles = []
    for q_name, fp_name in unique_pairs:
        handles.append(
            full_q[q_name].register_forward_hook(make_out_hook(module_out_q, q_name))
        )
        handles.append(
            full_fp[fp_name].register_forward_hook(make_out_hook(module_out_fp, fp_name))
        )
    return module_out_q, module_out_fp, handles


def _accumulate_module_mse_totals(
    module_mse_totals,
    module_out_q,
    module_out_fp,
    unique_pairs,
    scope,
    layer_idx,
):
    for q_name, fp_name in unique_pairs:
        q_val = module_out_q.get(q_name, None)
        fp_val = module_out_fp.get(fp_name, None)
        if q_val is None or fp_val is None:
            continue
        diff_mod = q_val.float() - fp_val.float()
        key = _build_module_key(scope, layer_idx, q_name)
        stat = module_mse_totals.setdefault(key, {"sse": 0.0, "numel": 0})
        stat["sse"] += diff_mod.pow(2).sum().item()
        stat["numel"] += diff_mod.numel()


def _finalize_module_mse_stats(module_mse_totals):
    module_mse = []
    for key in sorted(module_mse_totals.keys()):
        stat = module_mse_totals[key]
        module_mse.append(
            {
                "module_key": key,
                "mse": float(stat["sse"]) / max(int(stat["numel"]), 1),
                "sse": float(stat["sse"]),
                "numel": int(stat["numel"]),
            }
        )
    return module_mse


def _pack_quant_outputs(
    inps,
    layer_outputs,
    layer_mse,
    module_mse,
    collect_layer_outputs,
    collect_layer_mse,
    collect_module_mse,
):
    if collect_layer_outputs and collect_layer_mse and collect_module_mse:
        return inps, layer_outputs, layer_mse, module_mse
    if collect_layer_outputs and collect_module_mse:
        return inps, layer_outputs, module_mse
    if collect_layer_mse and collect_module_mse:
        return inps, layer_mse, module_mse
    if collect_layer_outputs and collect_layer_mse:
        return inps, layer_outputs, layer_mse
    if collect_layer_outputs:
        return inps, layer_outputs
    if collect_layer_mse:
        return inps, layer_mse
    if collect_module_mse:
        return inps, module_mse
    return inps


@torch.no_grad()
def build_int_quant_pack_from_weight(weight, n_bit=4, q_group_size=128, zero_point=True):
    w = weight.detach().float()
    org_shape = w.shape
    if q_group_size > 0:
        if org_shape[-1] % q_group_size != 0:
            raise ValueError(
                f"Weight last dim {org_shape[-1]} is not divisible by q_group_size={q_group_size}."
            )
        w2 = w.reshape(-1, q_group_size)
    else:
        w2 = w.reshape(w.shape[0], -1)
    if w2.dim() != 2:
        raise ValueError(f"Expected 2D reshaped weight, got shape={tuple(w2.shape)}")

    if zero_point:
        max_val = w2.amax(dim=1, keepdim=True)
        min_val = w2.amin(dim=1, keepdim=True)
        max_int = 2**int(n_bit) - 1
        min_int = 0
        scale = (max_val - min_val).clamp(min=1e-5) / float(max_int)
        zero = (-torch.round(min_val / scale)).clamp_(min_int, max_int)
    else:
        max_val = w2.abs().amax(dim=1, keepdim=True).clamp(min=1e-5)
        max_int = 2 ** (int(n_bit) - 1) - 1
        min_int = -(2 ** (int(n_bit) - 1))
        scale = max_val / float(max_int)
        zero = torch.zeros_like(scale)

    q = torch.clamp(torch.round(w2 / scale) + zero, min_int, max_int).to(torch.int16)
    deq = ((q.float() - zero) * scale).reshape(org_shape).to(weight.dtype)
    pack = {
        "qweight_int": q.reshape(org_shape).cpu(),
        "scale": scale.view(org_shape[0], -1).cpu(),
        "zero": zero.view(org_shape[0], -1).cpu(),
        "n_bit": int(n_bit),
        "q_group_size": int(q_group_size),
        "zero_point": bool(zero_point),
        "shape": tuple(org_shape),
    }
    return pack, deq


@torch.no_grad()
def dequantize_from_int_quant_pack(pack, dtype=torch.float32, device="cpu"):
    q = pack["qweight_int"].to(device=device).float()
    shape = tuple(pack["shape"])
    q_group_size = int(pack["q_group_size"])
    if q_group_size > 0:
        q2 = q.reshape(-1, q_group_size)
    else:
        q2 = q.reshape(shape[0], -1)
    scale = pack["scale"].to(device=device).reshape(-1, 1).float()
    zero = pack["zero"].to(device=device).reshape(-1, 1).float()
    w = ((q2 - zero) * scale).reshape(shape)
    return w.to(dtype=dtype)


@torch.no_grad()
def export_whisper_int_quant_bundle(model, n_bit=4, q_group_size=128, zero_point=True):
    bundle = {}
    encoder = model.model.encoder
    for i, layer in enumerate(encoder.layers):
        full = find_layers(layer)
        for name, mod in full.items():
            pack, _ = build_int_quant_pack_from_weight(
                mod.weight.data, n_bit=n_bit, q_group_size=q_group_size, zero_point=zero_point
            )
            bundle[f"enc.layer{i}.{name}"] = pack

    decoder = model.model.decoder
    for i, layer in enumerate(decoder.layers):
        full = find_layers(layer)
        for name, mod in full.items():
            pack, _ = build_int_quant_pack_from_weight(
                mod.weight.data, n_bit=n_bit, q_group_size=q_group_size, zero_point=zero_point
            )
            bundle[f"dec.layer{i}.{name}"] = pack
    return bundle


@torch.no_grad()
def apply_whisper_int_quant_bundle(model, bundle_payload):
    layers = bundle_payload["layers"] if isinstance(bundle_payload, dict) and "layers" in bundle_payload else bundle_payload
    n_applied = 0
    missing = []

    def _apply_to_scope(scope_name, layer_idx, mod_name, pack):
        nonlocal n_applied
        nonlocal missing
        scope = model.model.encoder if scope_name == "enc" else model.model.decoder
        if layer_idx < 0 or layer_idx >= len(scope.layers):
            missing.append(f"{scope_name}.layer{layer_idx}.{mod_name}")
            return
        layer = scope.layers[layer_idx]
        full = find_layers(layer)
        mod = full.get(mod_name, None)
        if mod is None:
            resolved = _resolve_layer_group_if_present(full, [mod_name])
            if resolved is not None:
                mod = full.get(resolved[0], None)
        if mod is None or not hasattr(mod, "weight"):
            missing.append(f"{scope_name}.layer{layer_idx}.{mod_name}")
            return
        w = dequantize_from_int_quant_pack(
            pack,
            dtype=mod.weight.data.dtype,
            device=mod.weight.data.device,
        )
        mod.weight.data.copy_(w)
        n_applied += 1

    for key, pack in layers.items():
        parts = key.split(".")
        if len(parts) < 3:
            continue
        scope_name = parts[0]
        layer_tok = parts[1]
        if scope_name not in ("enc", "dec") or not layer_tok.startswith("layer"):
            continue
        layer_idx = int(layer_tok[len("layer"):])
        mod_name = ".".join(parts[2:])
        _apply_to_scope(scope_name, layer_idx, mod_name, pack)

    return {"applied": n_applied, "missing": missing}


@torch.no_grad()
def export_qwen_int_quant_bundle(model, n_bit=4, q_group_size=128, zero_point=True):
    bundle = {}
    core = getattr(model, "model", model)
    thinker = getattr(core, "thinker", None)
    if thinker is None:
        thinker = getattr(model, "thinker", None)
    if thinker is None:
        return bundle

    audio_tower = getattr(thinker, "audio_tower", None)
    if audio_tower is not None and hasattr(audio_tower, "layers"):
        for i, layer in enumerate(audio_tower.layers):
            full = find_layers(layer)
            for name, mod in full.items():
                pack, _ = build_int_quant_pack_from_weight(
                    mod.weight.data, n_bit=n_bit, q_group_size=q_group_size, zero_point=zero_point
                )
                bundle[f"qwen_audio.layer{i}.{name}"] = pack

    text_model = getattr(thinker, "model", None)
    if text_model is not None and hasattr(text_model, "layers"):
        for i, layer in enumerate(text_model.layers):
            full = find_layers(layer)
            for name, mod in full.items():
                pack, _ = build_int_quant_pack_from_weight(
                    mod.weight.data, n_bit=n_bit, q_group_size=q_group_size, zero_point=zero_point
                )
                bundle[f"qwen_text.layer{i}.{name}"] = pack
    return bundle


@torch.no_grad()
def apply_qwen_int_quant_bundle(model, bundle_payload):
    layers = bundle_payload["layers"] if isinstance(bundle_payload, dict) and "layers" in bundle_payload else bundle_payload
    n_applied = 0
    missing = []

    core = getattr(model, "model", model)
    thinker = getattr(core, "thinker", None)
    if thinker is None:
        thinker = getattr(model, "thinker", None)
    if thinker is None:
        return {"applied": 0, "missing": ["qwen.thinker"]}

    def _scope_from_name(scope_name):
        if scope_name == "qwen_audio":
            return getattr(thinker, "audio_tower", None)
        if scope_name == "qwen_text":
            return getattr(thinker, "model", None)
        return None

    def _apply(scope_name, layer_idx, mod_name, pack):
        nonlocal n_applied
        nonlocal missing
        scope = _scope_from_name(scope_name)
        if scope is None or not hasattr(scope, "layers"):
            missing.append(f"{scope_name}.layer{layer_idx}.{mod_name}")
            return
        if layer_idx < 0 or layer_idx >= len(scope.layers):
            missing.append(f"{scope_name}.layer{layer_idx}.{mod_name}")
            return
        layer = scope.layers[layer_idx]
        full = find_layers(layer)
        mod = full.get(mod_name, None)
        if mod is None:
            resolved = _resolve_layer_group_if_present(full, [mod_name])
            if resolved is not None:
                mod = full.get(resolved[0], None)
        if mod is None or not hasattr(mod, "weight"):
            missing.append(f"{scope_name}.layer{layer_idx}.{mod_name}")
            return
        w = dequantize_from_int_quant_pack(
            pack,
            dtype=mod.weight.data.dtype,
            device=mod.weight.data.device,
        )
        mod.weight.data.copy_(w)
        n_applied += 1

    for key, pack in layers.items():
        parts = key.split(".")
        if len(parts) < 3:
            continue
        scope_name = parts[0]
        layer_tok = parts[1]
        if scope_name not in ("qwen_audio", "qwen_text") or not layer_tok.startswith("layer"):
            continue
        layer_idx = int(layer_tok[len("layer"):])
        mod_name = ".".join(parts[2:])
        _apply(scope_name, layer_idx, mod_name, pack)

    return {"applied": n_applied, "missing": missing}
