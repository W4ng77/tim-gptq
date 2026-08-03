import copy
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm

from frame_weighting import (
    frame_weights_apply_to_group,
    record_frame_weight_statistics,
)
from gptq import Helper
from modelutils import find_layers
from quantization_utils import (
    _move_layer_kwargs,
    _record_fp16_protected_module,
    _record_qep_gate_selection,
    _record_rotation_module,
    _resolve_layer_group_if_present,
    _should_apply_qep,
    _should_protect_ffn_output,
    _squeeze_batch,
    _to_cpu_detached,
    get_encoder_input,
    pseudo_quantize_tensor,
)


def _resolve_layer_group(full, names):
    resolved = _resolve_layer_group_if_present(full, names)
    if resolved is None:
        raise KeyError(f"Layer group not found: {names}")
    return resolved


def _build_dump_cfg(args, mode_tag):
    dump_dir = getattr(args, "dump_layer_tensors_dir", "")
    if not dump_dir:
        return None
    return {
        "base_dir": dump_dir,
        "mode_tag": str(mode_tag),
        "sample_idx": int(getattr(args, "dump_tensor_sample_idx", 0)),
    }


def _sanitize_name(name):
    return name.replace(".", "__")


def _maybe_linear_wx(x, mod):
    if (x is None) or (not isinstance(mod, nn.Linear)):
        return None
    try:
        return F.linear(x.float(), mod.weight.detach().float(), bias=None).detach().cpu()
    except Exception:
        return None


def _collect_module_io(layer, module_names, x, sample_kwargs, sample_idx, run_layer):
    captures = {}
    handles = []

    def _make_hook(name):
        def _hook(_m, inp, out):
            x_in = inp[0] if (isinstance(inp, (tuple, list)) and len(inp) > 0) else None
            y_out = out[0] if isinstance(out, tuple) else out
            captures[name] = {
                "x": None if x_in is None else x_in.detach().cpu(),
                "activation": None if y_out is None else y_out.detach().cpu(),
            }

        return _hook

    named = dict(layer.named_modules())
    for n in module_names:
        if n in named:
            handles.append(named[n].register_forward_hook(_make_hook(n)))
    try:
        _ = run_layer(layer, x, sample_kwargs, sample_idx)
    finally:
        for h in handles:
            h.remove()
    return captures


def _save_module_dump(
    dump_cfg,
    layer_key_prefix,
    layer_idx,
    module_name,
    payload,
):
    out_dir = os.path.join(
        dump_cfg["base_dir"],
        dump_cfg["mode_tag"],
        f"{layer_key_prefix}.{layer_idx}",
    )
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{_sanitize_name(module_name)}.pt")
    torch.save(payload, out_path)


def _awq_stack_inputs(input_samples, max_tokens=512):
    rows = []
    for x in input_samples:
        if x is None:
            continue
        t = x.detach()
        if t.dim() == 1:
            t = t.unsqueeze(0)
        t = t.reshape(-1, t.shape[-1]).float().cpu()
        if t.numel() > 0:
            rows.append(t)
    if len(rows) == 0:
        return None
    x = torch.cat(rows, dim=0)
    if x.shape[0] > max_tokens:
        idx = torch.linspace(0, x.shape[0] - 1, steps=max_tokens).long()
        x = x.index_select(0, idx)
    return x


def _awq_search_scale_weight(weight, input_feat, n_bit, q_group_size, zero_point, n_grid=20):
    if input_feat is None or input_feat.numel() == 0:
        return pseudo_quantize_tensor(
            weight,
            n_bit=n_bit,
            q_group_size=q_group_size,
            zero_point=zero_point,
        )

    dev = weight.device
    x = input_feat.to(device=dev, dtype=torch.float32)
    w = weight.float()

    with torch.no_grad():
        org_out = x.matmul(w.t())
        x_max = x.abs().mean(dim=0).clamp(min=1e-4)

        best_err = float("inf")
        best_w = None
        for i in range(n_grid):
            ratio = float(i) / float(max(n_grid, 1))
            scales = x_max.pow(ratio).clamp(min=1e-4)
            scales = scales / torch.sqrt(scales.max() * scales.min())
            w_scaled = w * scales.view(1, -1)
            w_q = pseudo_quantize_tensor(
                w_scaled,
                n_bit=n_bit,
                q_group_size=q_group_size,
                zero_point=zero_point,
            )
            w_q = w_q / scales.view(1, -1)
            out = x.matmul(w_q.t())
            err = (org_out - out).pow(2).mean().item()
            if err < best_err:
                best_err = err
                best_w = w_q

    return best_w if best_w is not None else w


def _awq_auto_clip_max(weight, input_feat, n_bit, q_group_size, zero_point, n_grid=20, max_shrink=0.5):
    if input_feat is None or input_feat.numel() == 0:
        return None

    w = weight.float()
    group_size = int(q_group_size) if int(q_group_size) > 0 else int(w.shape[1])
    if w.shape[1] % group_size != 0:
        group_size = int(w.shape[1])

    x = input_feat.float().reshape(-1, w.shape[1])
    if x.shape[0] > 512:
        idx = torch.linspace(0, x.shape[0] - 1, steps=512).long()
        x = x.index_select(0, idx)
    x = x.reshape(1, x.shape[0], -1, group_size).to(w.device)

    w_view = w.reshape(w.shape[0], 1, -1, group_size)
    org_max = w_view.abs().amax(dim=-1, keepdim=True)
    best_max = org_max.clone()
    best_err = torch.full_like(org_max, 1e9)
    org_out = (x * w_view).sum(dim=-1)

    oc_bs = 64
    for start in range(0, w_view.shape[0], oc_bs):
        end = min(start + oc_bs, w_view.shape[0])
        w_blk = w_view[start:end]
        org_blk = org_out[start:end]
        org_max_blk = org_max[start:end]
        best_max_blk = best_max[start:end]
        best_err_blk = best_err[start:end]
        for i in range(int(max_shrink * n_grid)):
            max_val = org_max_blk * (1 - float(i) / float(n_grid))
            min_val = -max_val
            w_clipped = torch.clamp(w_blk, min_val, max_val)
            # The clipping search uses a 4-D broadcast-friendly view; the
            # shared fake quantizer deliberately accepts matrices only.
            w_q = pseudo_quantize_tensor(
                w_clipped.reshape(-1, group_size),
                n_bit=n_bit,
                q_group_size=group_size,
                zero_point=zero_point,
            ).reshape_as(w_clipped)
            cur_out = (x * w_q).sum(dim=-1)
            err = (cur_out - org_blk).pow(2).mean(dim=1).view_as(best_err_blk)
            better = err < best_err_blk
            best_err_blk[better] = err[better]
            best_max_blk[better] = max_val[better]
        best_max[start:end] = best_max_blk

    return best_max.squeeze(1)


def _awq_should_skip_clip(module_name):
    if not module_name:
        return False
    return any(tok in module_name for tok in ["q_", "k_", "query", "key", "Wqkv"])


def _group_wbits(args, layer_key_prefix, layer_idx, num_layers, resolved_names):
    bits = int(args.wbits)
    if (
        layer_key_prefix == "dec"
        and resolved_names
        and resolved_names[0].startswith("encoder_attn.")
        and int(getattr(args, "cross_attn_wbits", 0)) > 0
    ):
        bits = int(args.cross_attn_wbits)
    tail_layers = int(getattr(args, "encoder_tail_layers", 0))
    tail_bits = int(getattr(args, "encoder_tail_wbits", 0))
    promoted = set(getattr(args, "encoder_promote_layer_indices", ()))
    if (
        layer_key_prefix == "enc"
        and tail_bits > 0
        and (
            (tail_layers > 0 and layer_idx >= max(0, num_layers - tail_layers))
            or layer_idx in promoted
        )
    ):
        bits = tail_bits
    return bits


def _record_group_bits(args, modules, bits):
    stats = getattr(args, "_bit_allocation_stats", None)
    if stats is None:
        stats = {"weight_numel_by_bits": {}}
        args._bit_allocation_stats = stats
    key = str(int(bits))
    stats["weight_numel_by_bits"][key] = stats["weight_numel_by_bits"].get(key, 0) + sum(
        int(module.weight.numel()) for module in modules
    )


@torch.no_grad()
def awq_quantize_linear_module(mod, input_samples, wbits, groupsize, zero_point=True, module_name=""):
    if (not isinstance(mod, nn.Linear)) or (not hasattr(mod, "weight")):
        return False

    q_config = {"q_group_size": int(groupsize)}
    q_group_size = int(q_config["q_group_size"])

    x = _awq_stack_inputs(input_samples, max_tokens=512)
    if x is None:
        mod.weight.data = pseudo_quantize_tensor(
            mod.weight.data,
            n_bit=wbits,
            q_group_size=q_group_size,
            zero_point=zero_point,
        ).to(mod.weight.data.dtype)
        return False

    w = mod.weight.data.float()
    w = _awq_search_scale_weight(
        w,
        x,
        n_bit=wbits,
        q_group_size=q_group_size,
        zero_point=zero_point,
    )
    clip_max = None
    if not _awq_should_skip_clip(module_name):
        clip_max = _awq_auto_clip_max(
            w,
            x,
            n_bit=wbits,
            q_group_size=q_group_size,
            zero_point=zero_point,
        )
    if clip_max is not None:
        wv = w.reshape(clip_max.shape[0], clip_max.shape[1], -1)
        clip_max = clip_max.to(wv.device)
        wv = torch.clamp(wv, -clip_max, clip_max)
        w = wv.reshape_as(w)

    w_q = pseudo_quantize_tensor(
        w,
        n_bit=wbits,
        q_group_size=q_group_size,
        zero_point=zero_point,
    )
    mod.weight.data.copy_(w_q.to(mod.weight.data.dtype))
    return True


# -----------------------------
# 4) Pass-1: build encoder inps + kwargs (same trick as your code)
# -----------------------------
@torch.no_grad()
def build_encoder_inps(model, dev, calibration_data):
    use_cache = model.config.use_cache
    model.config.use_cache = False

    encoder = model.model.encoder
    encoder = encoder.to(dev)
    layers = encoder.layers

    dtype = next(iter(model.parameters())).dtype
    nsamples = len(calibration_data)
    cache = {"layer_kwargs_list": []}
    inps_list = []

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, *args, **kwargs):
            inps_list.append(_squeeze_batch(inp.detach().cpu()))
            cache["layer_kwargs_list"].append(
                {k: _to_cpu_detached(v) for k, v in kwargs.items()}
            )
            raise ValueError

    layers[0] = Catcher(layers[0])

    for batch in calibration_data:
        try:
            x = get_encoder_input(batch).to(dev).to(dtype)
            encoder(x)
        except ValueError:
            pass

    # layers[0] = layers[0].module
    # layers[0] = layers[0].cpu()
    # encoder = encoder.cpu()
    encoder.layers[0] = layers[0].module  # unwrap via encoder directly
    encoder = encoder.cpu()
    torch.cuda.empty_cache()

    if not inps_list:
        raise RuntimeError("Failed to capture encoder inputs for calibration.")

    norm_list = []
    lengths = []
    for t in inps_list:
        t = _squeeze_batch(t)
        if t.dim() != 2:
            raise RuntimeError(f"Unexpected encoder input shape: {tuple(t.shape)}")
        lengths.append(t.shape[0])
        norm_list.append(t)

    max_len = max(t.shape[0] for t in norm_list)
    hidden_size = norm_list[0].shape[1]

    inps = torch.zeros((len(norm_list), max_len, hidden_size), dtype=dtype, device=dev)
    for i, t in enumerate(norm_list):
        inps[i, : t.shape[0]] = t.to(device=dev, dtype=dtype)

    model.config.use_cache = use_cache
    return inps, cache["layer_kwargs_list"], lengths


# -----------------------------
# 6) Pass-1: build decoder inps + kwargs (same trick as your code)
# -----------------------------
@torch.no_grad()
def build_decoder_inps(model, dev, calibration_data, encoder_outputs):
    use_cache = model.config.use_cache
    model.config.use_cache = False

    decoder = model.model.decoder
    decoder = decoder.to(dev)
    layers = decoder.layers

    dtype = next(iter(model.parameters())).dtype
    cache = {"layer_kwargs_list": []}
    inps_list = []

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, *args, **kwargs):
            inps_list.append(_squeeze_batch(inp.detach().cpu()))
            cache["layer_kwargs_list"].append(
                {k: _to_cpu_detached(v) for k, v in kwargs.items()}
            )
            raise ValueError

    layers[0] = Catcher(layers[0])

    for idx, batch in enumerate(calibration_data):
        try:
            dec_ids = batch["decoder_input_ids"].to(dev)
            enc_hidden = encoder_outputs[idx].unsqueeze(0).to(dev)
            decoder(input_ids=dec_ids, encoder_hidden_states=enc_hidden)
        except ValueError:
            pass

    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()
    decoder = decoder.cpu()
    torch.cuda.empty_cache()

    if not inps_list:
        raise RuntimeError("Failed to capture decoder inputs for calibration.")

    norm_list = []
    lengths = []
    for t in inps_list:
        t = _squeeze_batch(t)
        if t.dim() != 2:
            raise RuntimeError(f"Unexpected decoder input shape: {tuple(t.shape)}")
        lengths.append(t.shape[0])
        norm_list.append(t)

    max_len = max(t.shape[0] for t in norm_list)
    hidden_size = norm_list[0].shape[1]
    inps = torch.zeros((len(norm_list), max_len, hidden_size), dtype=dtype, device=dev)
    for i, t in enumerate(norm_list):
        inps[i, : t.shape[0]] = t.to(device=dev, dtype=dtype)

    model.config.use_cache = use_cache
    return inps, cache["layer_kwargs_list"], lengths


# -----------------------------
# 9) Pass-2: sequential quantization
# -----------------------------
@torch.no_grad()
def _quantize_stack_with_alpha(
    layers,
    inps,
    layer_kwargs_list,
    lengths,
    args,
    dev,
    dtype,
    sequential,
    layer_key_prefix,
    run_layer,
    gptq_nsamples=None,
    collect_layer_outputs=False,
    output_sample_idx=0,
    collect_layer_mse=False,
    error_channel_weight=None,
    alpha_by_layer=None,
    plot_stats=False,
):
    inps_true = inps.clone()
    nsamples = inps.shape[0]
    nsamples_gptq = nsamples if gptq_nsamples is None else min(gptq_nsamples, nsamples)
    layer_outputs = [] if collect_layer_outputs else None
    layer_mse = [] if collect_layer_mse else None


    def get_kwargs(j):
        if not layer_kwargs_list:
            return {}
        kw = layer_kwargs_list[min(j, len(layer_kwargs_list) - 1)]
        return _move_layer_kwargs(kw, dev, dtype)

    for i in tqdm.tqdm(
        range(len(layers)), desc=f"[PASS2] Quant {layer_key_prefix} ({args.method})"
    ):
        layer = layers[i].to(dev)
        layer_true = copy.deepcopy(layers[i]).to(dev)
        layer_alpha = args.perccorr
        if alpha_by_layer is not None:
            layer_alpha = float(alpha_by_layer.get(f"{layer_key_prefix}.{i}", layer_alpha))

        full = find_layers(layer)
        full_true = find_layers(layer_true)

        for names in sequential:
            resolved_names = _resolve_layer_group(full, names)
            resolved_names_true = _resolve_layer_group(full_true, names)
            subset = {n: full[n] for n in resolved_names}
            subset_true = {n: full_true[n] for n in resolved_names_true}
            if all(
                _should_protect_ffn_output(
                    args,
                    name,
                    layer_idx=i,
                    total_layers=len(layers),
                )
                for name in resolved_names
            ):
                for name, mod in subset.items():
                    _record_fp16_protected_module(
                        args,
                        f"{layer_key_prefix}.layer{i}.{name}",
                        mod,
                    )
                continue
            group_wbits = _group_wbits(
                args,
                layer_key_prefix,
                i,
                len(layers),
                resolved_names,
            )
            _record_group_bits(args, subset.values(), group_wbits)

            hook_latest = {}
            hook_latest_true = {}
            hook_inputs = {}

            def make_hook(latest, name, history=None):
                def hook(m, inp, out):
                    latest[name] = inp[0].detach().clone()
                    if history is not None:
                        history.setdefault(name, []).append(inp[0].detach().cpu())

                return hook

            handles = []
            for name, mod in subset.items():
                hist = hook_inputs if args.method == "awq" else None
                handles.append(mod.register_forward_hook(make_hook(hook_latest, name, hist)))
            if args.method == "gptq":
                for name, mod in subset_true.items():
                    handles.append(mod.register_forward_hook(make_hook(hook_latest_true, name)))


            helper = None

            if args.method == "gptq":
                helper = Helper(subset[resolved_names[0]])
                apply_qep = _should_apply_qep(args, resolved_names[0])
                # GPTAQ (arXiv:2504.02692) needs the FP-stream cross Gram
                # H_delta for every quantized group, via the same dual-stream
                # statistics QEP already collects.
                use_dual_stream = apply_qep or bool(getattr(args, "gptaq", False))
                frame_weights = getattr(args, "_frame_token_weights", None)
                frame_weighted_group = (
                    frame_weights is not None
                    and frame_weights_apply_to_group(
                        layer_key_prefix,
                        resolved_names,
                    )
                )
                if frame_weighted_group and use_dual_stream:
                    # The CLI validates this earlier; keep a hard stop so the
                    # dual-stream Gram can never silently drop the weights.
                    raise RuntimeError(
                        "Frame weighting is not supported on the dual-stream "
                        "(QEP/GPTAQ) statistics path: Helper.add_batch_qep "
                        "does not accept token_weights."
                    )
                for j in range(nsamples_gptq):
                    seq_len = lengths[j]
                    sample_kwargs = get_kwargs(j)
                    _ = run_layer(
                        layer,
                        inps[j, :seq_len].unsqueeze(0),
                        sample_kwargs,
                        j,
                    )
                    if use_dual_stream:
                        _ = run_layer(
                            layer_true,
                            inps_true[j, :seq_len].unsqueeze(0),
                            sample_kwargs,
                            j,
                        )
                        helper.add_batch_qep(
                            hook_latest[resolved_names[0]],
                            hook_latest_true[resolved_names_true[0]],
                        )
                    else:
                        token_weights = None
                        if frame_weighted_group:
                            sample_frame_weights = frame_weights[j]
                            if sample_frame_weights is not None:
                                token_weights = sample_frame_weights
                                record_frame_weight_statistics(
                                    args,
                                    (
                                        f"{layer_key_prefix}.layer{i}."
                                        f"{resolved_names[0]}"
                                    ),
                                    sample_frame_weights,
                                )
                        if token_weights is None:
                            sequence_weights = getattr(
                                args,
                                "_sequence_token_weights",
                                None,
                            )
                            if sequence_weights is not None:
                                module_key = (
                                    f"{layer_key_prefix}.layer{i}."
                                    f"{resolved_names[0]}"
                                )
                                module_weights = sequence_weights.get(module_key)
                                if module_weights is not None:
                                    token_weights = module_weights[j]
                        helper.add_batch(
                            hook_latest[resolved_names[0]],
                            token_weights=token_weights,
                        )
            elif args.method == "awq":
                for j in range(nsamples_gptq):
                    seq_len = lengths[j]
                    sample_kwargs = get_kwargs(j)
                    _ = run_layer(layer, inps[j, :seq_len].unsqueeze(0), sample_kwargs, j)

            for h in handles:
                h.remove()

            for name, mod in subset.items():
                if args.method == "rtn":
                    mod.weight.data = pseudo_quantize_tensor(
                        mod.weight.data, n_bit=group_wbits, q_group_size=args.groupsize
                    )
                elif args.method == "gptq":
                    apply_qep = _should_apply_qep(args, name)
                    if bool(getattr(args, "qep_gate", False)) and apply_qep:
                        W_gptq, selected_alpha, candidate_scores = (
                            helper.run_gptq_qep_candidates(
                                mod,
                                candidates=args.qep_gate_candidates,
                                percdampqep=args.percdampqep,
                                percdamp=args.percdamp,
                                wbits=group_wbits,
                                groupsize=args.groupsize,
                                actorder=args.act_order,
                            )
                        )
                        module_key = f"{layer_key_prefix}.layer{i}.{name}"
                        _record_qep_gate_selection(
                            args,
                            module_key,
                            selected_alpha,
                            candidate_scores,
                        )
                    else:
                        if apply_qep:
                            try:
                                helper.run_weight_correct(
                                    mod,
                                    percdamp=args.percdampqep,
                                    perccorr=layer_alpha,
                                )
                            except RuntimeError as e:
                                if "cholesky" in str(e).lower() and hasattr(helper, "H_q"):
                                    diag = torch.diag(helper.H_q).float()
                                    finite = torch.isfinite(diag)
                                    msg = (
                                        f"[QEP Cholesky Error] scope={layer_key_prefix} layer={i} "
                                        f"group={resolved_names} mod={name} "
                                        f"diag_min={diag[finite].min().item() if finite.any() else float('nan'):.6e} "
                                        f"diag_max={diag[finite].max().item() if finite.any() else float('nan'):.6e} "
                                        f"diag_mean={diag[finite].mean().item() if finite.any() else float('nan'):.6e} "
                                        f"diag_nonfinite={(~finite).sum().item()}"
                                    )
                                    print(msg, flush=True)
                                raise
                        rotate = str(getattr(args, "rotate", "none"))
                        module_key = f"{layer_key_prefix}.layer{i}.{name}"
                        W_gptq = helper.run_gptq(
                            mod,
                            percdamp=args.percdamp,
                            wbits=group_wbits,
                            groupsize=args.groupsize,
                            actorder=args.act_order,
                            return_W=True,
                            gptaq_alpha=(
                                float(getattr(args, "gptaq_alpha", 0.25))
                                if bool(getattr(args, "gptaq", False))
                                else None
                            ),
                            rotate=rotate,
                            rotation_seed=int(getattr(args, "rotation_seed", 0)),
                            rotation_tag=module_key,
                        )
                        if rotate != "none":
                            _record_rotation_module(
                                args,
                                module_key,
                                int(mod.weight.shape[1]),
                            )
                    mod.weight.data.copy_(W_gptq.to(mod.weight.data.dtype))
                elif args.method == "awq":
                    awq_quantize_linear_module(
                        mod,
                        hook_inputs.get(name, []),
                        wbits=group_wbits,
                        groupsize=args.groupsize,
                        zero_point=bool(getattr(args, "int_zero_point", True)),
                        module_name=name,
                    )
                else:
                    raise NotImplementedError("This script supports rtn/gptq/awq only.")

            if helper is not None:
                helper.free()


        for j in range(nsamples):
            seq_len = lengths[j]
            sample_kwargs = get_kwargs(j)
            sample_kwargs_true = get_kwargs(j)
            out = run_layer(layer, inps[j, :seq_len].unsqueeze(0), sample_kwargs, j)[0]
            out_true = run_layer(
                layer_true,
                inps_true[j, :seq_len].unsqueeze(0),
                sample_kwargs_true,
                j,
            )[0]
            inps[j, :seq_len] = out
            inps_true[j, :seq_len] = out_true
            if collect_layer_outputs and j == output_sample_idx:
                layer_outputs.append(out.detach().cpu())
            if collect_layer_mse:
                diff = out.float() - out_true.float()
                if j == 0:
                    sse = 0.0
                    weighted_sse = 0.0
                    count = 0
                sse += diff.pow(2).sum().item()
                if error_channel_weight is not None:
                    channel_weight = error_channel_weight.to(
                        device=diff.device,
                        dtype=diff.dtype,
                    )
                    weighted_sse += (diff.pow(2) * channel_weight).sum().item()
                count += diff.numel()

        if collect_layer_mse:
            layer_mse.append(
                {
                    "layer_idx": i,
                    "mse": sse / max(count, 1),
                    "context_weighted_mse": (
                        weighted_sse / max(count, 1)
                        if error_channel_weight is not None
                        else None
                    ),
                    "sse": sse,
                    "numel": count,
                }
            )

        layers[i] = layer.cpu()
        del layer, layer_true
        torch.cuda.empty_cache()

    if collect_layer_outputs and collect_layer_mse:
        return inps, layer_outputs, layer_mse
    if collect_layer_outputs:
        return inps, layer_outputs
    if collect_layer_mse:
        return inps, layer_mse
    return inps
