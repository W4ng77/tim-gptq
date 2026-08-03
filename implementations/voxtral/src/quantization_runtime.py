import copy
import io
import math
import sys
import warnings
from itertools import islice

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import tqdm

from dataset_sources import load_local_parquet_dataset
from quantization_utils import (
    _record_fp16_protected_module,
    _record_qep_gate_selection,
    _record_rotation_module,
    _resolve_layer_group_if_present,
    _should_apply_qep,
    _should_protect_ffn_output,
    pseudo_quantize_tensor,
)
from quantization_pipeline import _record_group_bits, awq_quantize_linear_module
from gptq import Helper


warnings.filterwarnings("ignore")

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _get_core_model(model):
    core = getattr(model, "model", None)
    if isinstance(core, nn.Module):
        return core
    if isinstance(model, nn.Module):
        return model
    return None


def _set_eval_mode(model):
    if hasattr(model, "eval"):
        try:
            model.eval()
            return model
        except Exception:
            pass
    core = _get_core_model(model)
    if isinstance(core, nn.Module):
        core.eval()
    return model


def _is_qwen_model_name(model_name):
    return "qwen" in str(model_name).lower()


def _is_qwen_model(args, model):
    if _is_qwen_model_name(getattr(args, "model", "")):
        return True
    cls_mod = model.__class__.__module__.lower()
    cls_name = model.__class__.__name__.lower()
    return ("qwen" in cls_mod) or ("qwen" in cls_name)


def _decode_audio_fallback(audio_info):
    audio, sr = sf.read(io.BytesIO(audio_info["bytes"]), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != 16000:
        duration = len(audio) / sr
        target_length = int(duration * 16000)
        audio = np.interp(
            np.linspace(0, len(audio) - 1, target_length),
            np.arange(len(audio)),
            audio,
        ).astype(np.float32)
    return audio


def _qwen_get_processor(model):
    core = _get_core_model(model)
    for owner in (core, model):
        if owner is None:
            continue
        proc = getattr(owner, "processor", None)
        if proc is not None:
            return proc
    return None


def _qwen_mask_transcript_labels(
    prompt_input_ids: torch.Tensor,
    full_input_ids: torch.Tensor,
) -> torch.Tensor:
    """Mask prompt positions while validating the teacher-forced token prefix."""
    if prompt_input_ids.ndim != 2 or full_input_ids.ndim != 2:
        raise ValueError("Qwen prompt and full input IDs must be rank-2 tensors.")
    if prompt_input_ids.shape[0] != full_input_ids.shape[0]:
        raise ValueError("Qwen prompt and full input batch sizes differ.")
    prompt_length = int(prompt_input_ids.shape[1])
    if int(full_input_ids.shape[1]) <= prompt_length:
        raise ValueError("Qwen teacher-forced input has no transcript tokens.")
    if not torch.equal(
        prompt_input_ids,
        full_input_ids[:, :prompt_length],
    ):
        raise ValueError(
            "Qwen full teacher-forced input does not preserve the prompt prefix."
        )
    labels = full_input_ids.clone()
    labels[:, :prompt_length] = -100
    if not (labels[:, prompt_length:] >= 0).all():
        raise ValueError("Qwen transcript labels contain invalid token IDs.")
    return labels


def _qwen_make_calibration_data(
    model,
    nsamples=128,
    seed=0,
    verbose=False,
    include_references=False,
    offset=0,
    include_labels=False,
):
    processor = _qwen_get_processor(model)
    if processor is None:
        raise RuntimeError(
            "Qwen calibration requires processor access. Expected model.model.processor."
        )

    dataset = load_local_parquet_dataset(
        "openslr/librispeech_asr",
        "clean",
        "train.100",
    )
    dataset = dataset.shuffle(
        seed=seed,
        buffer_size=max(10_000, (int(offset) + int(nsamples)) * 20),
    )

    text_prompt = getattr(processor, "audio_token", "<|audio|>")
    if include_labels:
        prompt_builder = getattr(model, "_build_text_prompt", None)
        if prompt_builder is None:
            raise RuntimeError(
                "Qwen sequence labels require the inference prompt builder."
            )
        text_prompt = prompt_builder("", "English")
    tokenizer = getattr(processor, "tokenizer", None)
    pad_token_id = getattr(tokenizer, "pad_token_id", None) if tokenizer is not None else None
    if pad_token_id is None:
        pad_token_id = getattr(tokenizer, "eos_token_id", None) if tokenizer is not None else None
    if pad_token_id is None:
        pad_token_id = 0

    calibs = []
    iterator = iter(dataset)
    try:
        samples = islice(iterator, int(offset), int(offset) + int(nsamples))
        for sample in tqdm.tqdm(
            samples,
            total=int(nsamples),
            desc="Loading Qwen calibration data",
            disable=not bool(verbose),
        ):
            audio_info = sample["audio"]
            audio = _decode_audio_fallback(audio_info)
            text = str(sample["text"])
            if include_labels:
                eos_token = getattr(tokenizer, "eos_token", "") or ""
                full_text = text_prompt + text + eos_token
                encoded_prompt = processor(
                    text=text_prompt,
                    audio=audio,
                    sampling_rate=16000,
                    return_tensors="pt",
                )
                encoded = processor(
                    text=full_text,
                    audio=audio,
                    sampling_rate=16000,
                    return_tensors="pt",
                )
            else:
                encoded_prompt = None
                encoded = processor(
                    text=text_prompt,
                    audio=audio,
                    sampling_rate=16000,
                    return_tensors="pt",
                )
            entry = {
                key: value
                for key, value in encoded.items()
                if torch.is_tensor(value)
            }
            if not entry:
                raise RuntimeError("Qwen processor output has no tensor fields.")
            entry["__pad_token_id__"] = int(pad_token_id)
            entry["__dataset_id__"] = str(sample["id"])
            if include_labels:
                entry["labels"] = _qwen_mask_transcript_labels(
                    encoded_prompt["input_ids"],
                    entry["input_ids"],
                )
            if include_references:
                entry["__audio__"] = audio
                entry["__text__"] = text
            calibs.append(entry)
    finally:
        close = getattr(iterator, "close", None)
        if close is not None:
            close()
    return calibs


def _qwen_get_thinker(model):
    core = _get_core_model(model)
    for owner in (core, model):
        if owner is None:
            continue
        thinker = getattr(owner, "thinker", None)
        if thinker is not None:
            return thinker
    return None


def _qwen_get_module_by_name(root, name):
    cur = root
    for part in name.split("."):
        cur = getattr(cur, part)
    return cur


def _qwen_collect_sequence_token_weights(
    model,
    calibration_data,
    dev,
):
    """Collect per-token sequence-loss sensitivity for Qwen text blocks."""
    model = _move_model_to_cuda_if_possible(model)
    model = _set_eval_mode(model)
    thinker = _qwen_get_thinker(model)
    if thinker is None or not hasattr(thinker, "model"):
        raise RuntimeError("Qwen model has no text backbone.")
    layers = thinker.model.layers
    targets = []
    for index, layer in enumerate(layers):
        module = dict(layer.named_modules()).get("mlp.down_proj")
        if not isinstance(module, nn.Linear):
            raise TypeError(
                f"Qwen text layer {index} has no linear mlp.down_proj."
            )
        targets.append(module)

    dtype = next(iter(_get_core_model(model).parameters())).dtype
    by_layer = {
        f"thinker.model.layers.{index}": []
        for index in range(len(targets))
    }
    summaries = {
        key: {"minimum": float("inf"), "maximum": 0.0, "sum": 0.0, "count": 0}
        for key in by_layer
    }

    for sample in calibration_data:
        captured = {}
        handles = []

        def make_hook(index):
            def hook(_module, _inputs, output):
                output.retain_grad()
                captured[index] = output

            return hook

        for index, module in enumerate(targets):
            handles.append(module.register_forward_hook(make_hook(index)))
        try:
            thinker.zero_grad(set_to_none=True)
            inputs = {
                key: value.to(dev)
                for key, value in sample.items()
                if torch.is_tensor(value)
            }
            if "input_features" in inputs:
                inputs["input_features"] = inputs["input_features"].to(dtype)
            feature_mask = inputs.get("feature_attention_mask")
            audio_feature_lengths = (
                feature_mask.sum(dim=1).long()
                if feature_mask is not None
                else None
            )
            with torch.enable_grad():
                output = thinker(
                    input_ids=inputs["input_ids"],
                    input_features=inputs.get("input_features"),
                    attention_mask=inputs.get("attention_mask"),
                    feature_attention_mask=feature_mask,
                    audio_feature_lengths=audio_feature_lengths,
                    labels=inputs["labels"],
                    use_cache=False,
                )
                output.loss.backward()
            for index in range(len(targets)):
                if index not in captured or captured[index].grad is None:
                    raise RuntimeError(
                        f"Missing sequence gradient for Qwen text layer {index}."
                    )
                gradient = captured[index].grad.detach().float()
                weights = gradient.square().mean(dim=-1).reshape(-1)
                weights = weights / weights.mean().clamp_min(1e-12)
                weights = weights.clamp(min=1e-4, max=100.0).cpu()
                weights = weights / weights.mean().clamp_min(1e-12)
                key = f"thinker.model.layers.{index}"
                by_layer[key].append(weights)
                summary = summaries[key]
                summary["minimum"] = min(summary["minimum"], float(weights.min()))
                summary["maximum"] = max(summary["maximum"], float(weights.max()))
                summary["sum"] += float(weights.sum())
                summary["count"] += int(weights.numel())
        finally:
            for handle in handles:
                handle.remove()
            thinker.zero_grad(set_to_none=True)

    for summary in summaries.values():
        summary["mean"] = summary.pop("sum") / max(summary["count"], 1)
    torch.cuda.empty_cache()
    return by_layer, summaries


def _module_device(module):
    for p in module.parameters():
        return p.device
    for b in module.buffers():
        return b.device
    return DEV


def _qwen_run_audio_layer(layer, hidden_states, layer_kwargs):
    out = layer(
        hidden_states,
        cu_seqlens=layer_kwargs["cu_seqlens"],
        attention_mask=layer_kwargs.get("attention_mask", None),
    )
    if isinstance(out, tuple):
        out = out[0]
    return out


def _qwen_run_text_layer(layer, hidden_states, layer_kwargs):
    out = layer(
        hidden_states,
        position_embeddings=layer_kwargs["position_embeddings"],
        attention_mask=layer_kwargs.get("attention_mask", None),
        position_ids=layer_kwargs.get("position_ids", None),
        past_key_values=layer_kwargs.get("past_key_values", None),
        use_cache=layer_kwargs.get("use_cache", False),
        cache_position=layer_kwargs.get("cache_position", None),
    )
    if isinstance(out, tuple):
        out = out[0]
    return out


class _QwenCatcherAudio(nn.Module):
    def __init__(self, layer, store):
        super().__init__()
        self.layer = layer
        self.store = store

    def forward(self, hidden_states, cu_seqlens, attention_mask=None, **kwargs):
        self.store["hidden_states"] = hidden_states.detach()
        self.store["cu_seqlens"] = cu_seqlens.detach() if torch.is_tensor(cu_seqlens) else cu_seqlens
        self.store["attention_mask"] = (
            attention_mask.detach() if torch.is_tensor(attention_mask) else attention_mask
        )
        raise RuntimeError("Qwen audio catcher stop")


class _QwenCatcherText(nn.Module):
    def __init__(self, layer, store):
        super().__init__()
        self.layer = layer
        self.store = store

    def forward(
        self,
        hidden_states,
        position_embeddings,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        use_cache=False,
        cache_position=None,
        **kwargs,
    ):
        self.store["hidden_states"] = hidden_states.detach()
        self.store["position_embeddings"] = tuple(
            x.detach() if torch.is_tensor(x) else x for x in position_embeddings
        )
        self.store["attention_mask"] = (
            attention_mask.detach() if torch.is_tensor(attention_mask) else attention_mask
        )
        self.store["position_ids"] = (
            position_ids.detach() if torch.is_tensor(position_ids) else position_ids
        )
        self.store["past_key_values"] = past_key_values
        self.store["use_cache"] = use_cache
        self.store["cache_position"] = (
            cache_position.detach() if torch.is_tensor(cache_position) else cache_position
        )
        raise RuntimeError("Qwen text catcher stop")


QWEN_AUDIO_GROUPS = [
    ["self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj"],
    ["self_attn.out_proj"],
    ["fc1"],
    ["fc2"],
]

QWEN_TEXT_GROUPS = [
    ["self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj"],
    ["self_attn.o_proj"],
    ["mlp.gate_proj", "mlp.up_proj"],
    ["mlp.down_proj"],
]


def _qwen_collect_audio_layer0_cache(calibration_data, model, dev, dtype, verbose=False):
    thinker = _qwen_get_thinker(model)
    if thinker is None or not hasattr(thinker, "audio_tower"):
        return []
    layers = thinker.audio_tower.layers
    if len(layers) == 0:
        return []
    orig = layers[0]
    audio_cache = []

    for sample in tqdm.tqdm(
        calibration_data,
        desc="Qwen cache audio layer0",
        disable=not bool(verbose),
    ):
        inputs = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in sample.items()}
        x = inputs["input_features"].to(dtype)
        fam = inputs.get("feature_attention_mask", None)
        if fam is None:
            feature_len = x.shape[-1]
        else:
            feature_len = int(fam.sum(dim=1).long()[0].item())
        input_feature = x[0]

        store = {}
        layers[0] = _QwenCatcherAudio(orig, store)
        try:
            with torch.no_grad():
                _ = thinker.audio_tower(
                    input_feature[:, :feature_len],
                    feature_lens=torch.tensor([feature_len], device=dev),
                )
        except RuntimeError as e:
            if "Qwen audio catcher stop" not in str(e):
                raise
        finally:
            layers[0] = orig

        audio_cache.append(
            {
                "hidden_states": store["hidden_states"].detach().cpu(),
                "cu_seqlens": store["cu_seqlens"].detach().cpu(),
                "attention_mask": None
                if store["attention_mask"] is None
                else store["attention_mask"].detach().cpu(),
            }
        )
    return audio_cache


def _qwen_collect_text_layer0_cache(calibration_data, model, dev, dtype, verbose=False):
    thinker = _qwen_get_thinker(model)
    if thinker is None or not hasattr(thinker, "model"):
        return []
    layers = thinker.model.layers
    if len(layers) == 0:
        return []
    orig = layers[0]
    text_cache = []

    for sample in tqdm.tqdm(
        calibration_data,
        desc="Qwen cache text layer0",
        disable=not bool(verbose),
    ):
        inputs = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in sample.items()}
        if "input_features" in inputs:
            inputs["input_features"] = inputs["input_features"].to(dtype)
        fam = inputs.get("feature_attention_mask", None)
        audio_feature_lengths = fam.sum(dim=1).long() if fam is not None else None

        store = {}
        layers[0] = _QwenCatcherText(orig, store)
        try:
            with torch.no_grad():
                _ = thinker(
                    input_ids=inputs["input_ids"],
                    input_features=inputs.get("input_features", None),
                    attention_mask=inputs.get("attention_mask", None),
                    feature_attention_mask=inputs.get("feature_attention_mask", None),
                    audio_feature_lengths=audio_feature_lengths,
                    use_cache=False,
                )
        except RuntimeError as e:
            if "Qwen text catcher stop" not in str(e):
                raise
        finally:
            layers[0] = orig

        text_cache.append(
            {
                "hidden_states": store["hidden_states"].detach().cpu(),
                "position_embeddings": tuple(
                    x.detach().cpu() if torch.is_tensor(x) else x
                    for x in store["position_embeddings"]
                ),
                "attention_mask": None
                if store["attention_mask"] is None
                else store["attention_mask"].detach().cpu(),
                "position_ids": None
                if store["position_ids"] is None
                else store["position_ids"].detach().cpu(),
                "past_key_values": None,
                "use_cache": False,
                "cache_position": None
                if store["cache_position"] is None
                else store["cache_position"].detach().cpu(),
            }
        )
    return text_cache


@torch.no_grad()
def _qwen_collect_module_stats_on_cached_layer(
    layer,
    fp_layer,
    module_name,
    cache_q,
    cache_fp,
    runner_fn,
    kw_builder_fn,
    use_qep,
    sequence_token_weights=None,
):
    if (
        sequence_token_weights is not None
        and len(sequence_token_weights) != len(cache_q)
    ):
        raise ValueError(
            "Qwen sequence-weight sample count does not match cached inputs: "
            f"{len(sequence_token_weights)} != {len(cache_q)}"
        )
    full = dict(layer.named_modules())
    if module_name not in full:
        return None, 0.0
    target_mod = full[module_name]
    helper = Helper(target_mod)

    fp_mod = None
    if fp_layer is not None:
        fp_full = dict(fp_layer.named_modules())
        fp_mod = fp_full.get(module_name, None)

    latest_inp = {"x": None}
    latest_inp_fp = {"x": None}
    sum_sq = 0.0
    numel = 0

    def hook_q(_, inp, __):
        latest_inp["x"] = inp[0].detach()

    def hook_fp(_, inp, __):
        latest_inp_fp["x"] = inp[0].detach()

    h_q = target_mod.register_forward_hook(hook_q)
    h_fp = fp_mod.register_forward_hook(hook_fp) if fp_mod is not None else None
    try:
        dev = _module_device(layer)
        for i in range(len(cache_q)):
            latest_inp["x"] = None
            latest_inp_fp["x"] = None

            sq = cache_q[i]
            x_q = sq["hidden_states"].to(dev)
            kw_q = kw_builder_fn(sq, dev)
            _ = runner_fn(layer, x_q, kw_q)

            x_in = latest_inp["x"]
            if x_in is None:
                continue

            if fp_layer is not None and cache_fp is not None and i < len(cache_fp):
                sf = cache_fp[i]
                x_fp = sf["hidden_states"].to(dev)
                kw_fp = kw_builder_fn(sf, dev)
                _ = runner_fn(fp_layer, x_fp, kw_fp)
                x_true = latest_inp_fp["x"]
                if x_true is None:
                    helper.add_batch(
                        x_in,
                        token_weights=(
                            sequence_token_weights[i]
                            if sequence_token_weights is not None
                            else None
                        ),
                    )
                    continue
                if use_qep:
                    helper.add_batch_qep(x_in, x_true)
                else:
                    helper.add_batch(
                        x_in,
                        token_weights=(
                            sequence_token_weights[i]
                            if sequence_token_weights is not None
                            else None
                        ),
                    )
                diff = x_true.float() - x_in.float()
                sum_sq += diff.pow(2).sum().item()
                numel += diff.numel()
            else:
                helper.add_batch(
                    x_in,
                    token_weights=(
                        sequence_token_weights[i]
                        if sequence_token_weights is not None
                        else None
                    ),
                )
    finally:
        h_q.remove()
        if h_fp is not None:
            h_fp.remove()

    return helper, (sum_sq / max(numel, 1))


def _qwen_audio_kw_from_cache(sample, dev):
    return {
        "cu_seqlens": sample["cu_seqlens"].to(dev),
        "attention_mask": None
        if sample["attention_mask"] is None
        else sample["attention_mask"].to(dev),
    }


def _qwen_text_kw_from_cache(sample, dev):
    return {
        "position_embeddings": tuple(
            t.to(dev) if torch.is_tensor(t) else t for t in sample["position_embeddings"]
        ),
        "attention_mask": None if sample["attention_mask"] is None else sample["attention_mask"].to(dev),
        "position_ids": None if sample["position_ids"] is None else sample["position_ids"].to(dev),
        "past_key_values": None,
        "use_cache": False,
        "cache_position": None if sample["cache_position"] is None else sample["cache_position"].to(dev),
    }


@torch.no_grad()
def _qwen_propagate_cached_layer(layer, cache, runner_fn, kw_builder_fn):
    next_cache = []
    dev = _module_device(layer)
    for sample in cache:
        x = sample["hidden_states"].to(dev)
        kw = kw_builder_fn(sample, dev)
        y = runner_fn(layer, x, kw)
        out = dict(sample)
        out["hidden_states"] = y.detach().cpu()
        next_cache.append(out)
    return next_cache


def _qwen_hidden_mse_from_caches(cache_q, cache_fp):
    if cache_q is None or cache_fp is None:
        return 0.0
    n = min(len(cache_q), len(cache_fp))
    if n <= 0:
        return 0.0
    sum_sq = 0.0
    numel = 0
    for i in range(n):
        xq = cache_q[i].get("hidden_states", None)
        xf = cache_fp[i].get("hidden_states", None)
        if (xq is None) or (xf is None):
            continue
        if xq.shape != xf.shape:
            min_len = min(xq.shape[0], xf.shape[0])
            xq = xq[:min_len]
            xf = xf[:min_len]
        diff = xf.float() - xq.float()
        sum_sq += diff.pow(2).sum().item()
        numel += diff.numel()
    return sum_sq / max(numel, 1)


def _qwen_group_key_from_module_name(name):
    parts = name.split(".")
    for i in range(len(parts) - 1):
        if parts[i] == "layers" and i + 1 < len(parts):
            return ".".join(parts[: i + 2])
    if len(parts) > 1:
        return ".".join(parts[:-1])
    return name


def _qwen_scope_includes_stack(scope, stack):
    if scope == "full":
        return stack in {"audio", "text"}
    if scope == "encoder":
        return stack == "audio"
    if scope in {"text-backbone", "text-attention", "text-ffn"}:
        return stack == "text"
    return False


def _qwen_scope_includes_group(scope, stack, group):
    if not _qwen_scope_includes_stack(scope, stack):
        return False
    if scope in {"full", "encoder", "text-backbone"}:
        return True
    group_name = group[0]
    if scope == "text-attention":
        return stack == "text" and group_name.startswith("self_attn.")
    if scope == "text-ffn":
        return stack == "text" and group_name.startswith("mlp.")
    return False


def _qwen_gptq_weight(
    helper,
    layer,
    local_name,
    full_name,
    args,
    alpha,
):
    module = _qwen_get_module_by_name(layer, local_name)
    apply_qep = _should_apply_qep(args, local_name)
    if bool(getattr(args, "qep_gate", False)) and apply_qep:
        weight, selected_alpha, candidate_scores = helper.run_gptq_qep_candidates(
            module,
            candidates=args.qep_gate_candidates,
            percdampqep=args.percdampqep,
            percdamp=args.percdamp,
            wbits=args.wbits,
            groupsize=args.groupsize,
            actorder=args.act_order,
        )
        _record_qep_gate_selection(
            args,
            full_name,
            selected_alpha,
            candidate_scores,
        )
        return weight
    if apply_qep:
        helper.run_weight_correct(
            module,
            percdamp=args.percdampqep,
            perccorr=alpha,
        )
    rotate = str(getattr(args, "rotate", "none"))
    weight = helper.run_gptq(
        module,
        percdamp=args.percdamp,
        wbits=args.wbits,
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
        rotation_tag=full_name,
    )
    if rotate != "none":
        _record_rotation_module(args, full_name, int(module.weight.shape[1]))
    return weight


@torch.no_grad()
def _qwen_quantize_cached_sequential(
    model,
    calibration_data,
    args,
    dev,
    alpha_by_module=None,
    collect_scores_only=False,
):
    model = _move_model_to_cuda_if_possible(model)
    model = _set_eval_mode(model)
    thinker = _qwen_get_thinker(model)
    if thinker is None:
        raise RuntimeError("Qwen model has no thinker module.")
    dtype = next(iter(_get_core_model(model).parameters())).dtype
    quant_scope = getattr(args, "quant_scope", "full")
    if quant_scope not in {
        "full",
        "encoder",
        "text-backbone",
        "text-attention",
        "text-ffn",
    }:
        raise ValueError(
            f"Unsupported Qwen quantization scope: {quant_scope!r}."
        )

    fp_model = None
    fp_thinker = None
    need_fp = bool(
        args.qep or args.dynamic_alpha or getattr(args, "gptaq", False)
    )
    if need_fp:
        fp_model = copy.deepcopy(model)
        fp_model = _move_model_to_cuda_if_possible(fp_model)
        fp_model = _set_eval_mode(fp_model)
        fp_thinker = _qwen_get_thinker(fp_model)

    module_scores = {}
    pbar = None
    if collect_scores_only:
        pbar_desc = "Qwen score"
    elif args.method == "awq":
        pbar_desc = "Qwen AWQ"
    elif args.method == "rtn":
        pbar_desc = "Qwen RTN"
    else:
        pbar_desc = "Qwen GPTQ"

    def _count_target_modules(layers, groups, stack):
        count = 0
        for lid in range(len(layers)):
            mods = dict(layers[lid].named_modules())
            for group in groups:
                if not _qwen_scope_includes_group(quant_scope, stack, group):
                    continue
                for local_name in group:
                    if local_name in mods and not _should_protect_ffn_output(
                        args,
                        local_name,
                        layer_idx=lid,
                        total_layers=len(layers),
                    ):
                        count += 1
        return count

    total_modules = 0
    if (
        _qwen_scope_includes_stack(quant_scope, "audio")
        and hasattr(thinker, "audio_tower")
        and hasattr(thinker.audio_tower, "layers")
    ):
        total_modules += _count_target_modules(
            thinker.audio_tower.layers,
            QWEN_AUDIO_GROUPS,
            "audio",
        )
    if (
        _qwen_scope_includes_stack(quant_scope, "text")
        and hasattr(thinker, "model")
        and hasattr(thinker.model, "layers")
    ):
        total_modules += _count_target_modules(
            thinker.model.layers,
            QWEN_TEXT_GROUPS,
            "text",
        )
    pbar = tqdm.tqdm(
        total=total_modules,
        desc=pbar_desc,
        disable=False,
        mininterval=0.5,
        dynamic_ncols=True,
        file=sys.stdout,
    )

    if (
        _qwen_scope_includes_stack(quant_scope, "audio")
        and hasattr(thinker, "audio_tower")
        and hasattr(thinker.audio_tower, "layers")
    ):
        cache_q = _qwen_collect_audio_layer0_cache(
            calibration_data, model, dev, dtype, verbose=bool(args.verbose)
        )
        cache_fp = None
        if need_fp and fp_model is not None:
            cache_fp = _qwen_collect_audio_layer0_cache(
                calibration_data, fp_model, dev, dtype, verbose=bool(args.verbose)
            )
        for lid in range(len(thinker.audio_tower.layers)):
            layer = thinker.audio_tower.layers[lid].to(dev).eval()
            fp_layer = None
            if fp_thinker is not None:
                fp_layer = fp_thinker.audio_tower.layers[lid].to(dev).eval()
            layer_local_names = []
            for group in QWEN_AUDIO_GROUPS:
                if not _qwen_scope_includes_group(quant_scope, "audio", group):
                    continue
                mods = dict(layer.named_modules())
                group_present = [n for n in group if n in mods]
                if len(group_present) == 0:
                    continue
                active_names = []
                for local_name in group_present:
                    if _should_protect_ffn_output(
                        args,
                        local_name,
                        layer_idx=lid,
                        total_layers=len(thinker.audio_tower.layers),
                    ):
                        full_name = (
                            f"thinker.audio_tower.layers.{lid}.{local_name}"
                        )
                        _record_fp16_protected_module(
                            args,
                            full_name,
                            mods[local_name],
                        )
                    else:
                        active_names.append(local_name)
                group_present = active_names
                if len(group_present) == 0:
                    continue
                if not collect_scores_only:
                    _record_group_bits(
                        args,
                        [mods[name] for name in group_present],
                        args.wbits,
                    )
                layer_local_names.extend(group_present)
                primary_name = group_present[0]
                if args.method == "awq":
                    input_samples = {n: [] for n in group_present}
                    handles = []
                    for local_name in group_present:
                        target_mod = _qwen_get_module_by_name(layer, local_name)

                        def _capture_awq_inp(_, inp, __, _n=local_name):
                            if isinstance(inp, (tuple, list)) and len(inp) > 0:
                                input_samples[_n].append(inp[0].detach().cpu())

                        handles.append(target_mod.register_forward_hook(_capture_awq_inp))
                    try:
                        for sample in cache_q:
                            x_q = sample["hidden_states"].to(dev)
                            kw_q = _qwen_audio_kw_from_cache(sample, dev)
                            _ = _qwen_run_audio_layer(layer, x_q, kw_q)
                    finally:
                        for h in handles:
                            h.remove()

                    for local_name in group_present:
                        awq_quantize_linear_module(
                            _qwen_get_module_by_name(layer, local_name),
                            input_samples.get(local_name, []),
                            wbits=args.wbits,
                            groupsize=args.groupsize,
                            zero_point=bool(getattr(args, "int_zero_point", True)),
                            module_name=local_name,
                        )
                        pbar.update(1)
                    continue
                helper, _ = _qwen_collect_module_stats_on_cached_layer(
                    layer,
                    fp_layer,
                    primary_name,
                    cache_q,
                    cache_fp,
                    _qwen_run_audio_layer,
                    _qwen_audio_kw_from_cache,
                    # GPTAQ needs the dual-stream cross Gram H_delta for
                    # every quantized group, like QEP.
                    use_qep=(
                        _should_apply_qep(args, primary_name)
                        or bool(getattr(args, "gptaq", False))
                    ),
                    sequence_token_weights=None,
                )
                if helper is None:
                    pbar.update(len(group_present))
                    continue
                for local_name in group_present:
                    full_name = f"thinker.audio_tower.layers.{lid}.{local_name}"
                    alpha = float(args.perccorr)
                    if alpha_by_module is not None:
                        alpha = float(alpha_by_module.get(full_name, alpha))
                    w_q = _qwen_gptq_weight(
                        helper,
                        layer,
                        local_name,
                        full_name,
                        args,
                        alpha,
                    )
                    _qwen_get_module_by_name(layer, local_name).weight.data.copy_(
                        w_q.to(_qwen_get_module_by_name(layer, local_name).weight.data.dtype)
                    )
                    pbar.update(1)
                helper.free()
            cache_q = _qwen_propagate_cached_layer(
                layer, cache_q, _qwen_run_audio_layer, _qwen_audio_kw_from_cache
            )
            if need_fp and fp_layer is not None and cache_fp is not None:
                cache_fp = _qwen_propagate_cached_layer(
                    fp_layer, cache_fp, _qwen_run_audio_layer, _qwen_audio_kw_from_cache
                )
            if collect_scores_only and (cache_fp is not None):
                layer_mse = float(_qwen_hidden_mse_from_caches(cache_q, cache_fp))
                for local_name in layer_local_names:
                    full_name = f"thinker.audio_tower.layers.{lid}.{local_name}"
                    module_scores[full_name] = layer_mse
            thinker.audio_tower.layers[lid] = layer.cpu()
            if fp_layer is not None:
                fp_thinker.audio_tower.layers[lid] = fp_layer.cpu()

    if (
        _qwen_scope_includes_stack(quant_scope, "text")
        and hasattr(thinker, "model")
        and hasattr(thinker.model, "layers")
    ):
        model = _move_model_to_cuda_if_possible(model)
        model = _set_eval_mode(model)
        thinker = _qwen_get_thinker(model)
        if fp_model is not None:
            fp_model = _move_model_to_cuda_if_possible(fp_model)
            fp_model = _set_eval_mode(fp_model)
            fp_thinker = _qwen_get_thinker(fp_model)

        cache_q = _qwen_collect_text_layer0_cache(
            calibration_data, model, dev, dtype, verbose=bool(args.verbose)
        )
        cache_fp = None
        if need_fp and fp_model is not None:
            cache_fp = _qwen_collect_text_layer0_cache(
                calibration_data, fp_model, dev, dtype, verbose=bool(args.verbose)
            )
        for lid in range(len(thinker.model.layers)):
            layer = thinker.model.layers[lid].to(dev).eval()
            fp_layer = None
            if fp_thinker is not None:
                fp_layer = fp_thinker.model.layers[lid].to(dev).eval()
            layer_local_names = []
            for group in QWEN_TEXT_GROUPS:
                if not _qwen_scope_includes_group(quant_scope, "text", group):
                    continue
                mods = dict(layer.named_modules())
                group_present = [n for n in group if n in mods]
                if len(group_present) == 0:
                    continue
                active_names = []
                for local_name in group_present:
                    if _should_protect_ffn_output(
                        args,
                        local_name,
                        layer_idx=lid,
                        total_layers=len(thinker.model.layers),
                    ):
                        full_name = f"thinker.model.layers.{lid}.{local_name}"
                        _record_fp16_protected_module(
                            args,
                            full_name,
                            mods[local_name],
                        )
                    else:
                        active_names.append(local_name)
                group_present = active_names
                if len(group_present) == 0:
                    continue
                if not collect_scores_only:
                    _record_group_bits(
                        args,
                        [mods[name] for name in group_present],
                        args.wbits,
                    )
                layer_local_names.extend(group_present)
                primary_name = group_present[0]
                if args.method == "awq":
                    input_samples = {n: [] for n in group_present}
                    handles = []
                    for local_name in group_present:
                        target_mod = _qwen_get_module_by_name(layer, local_name)

                        def _capture_awq_inp(_, inp, __, _n=local_name):
                            if isinstance(inp, (tuple, list)) and len(inp) > 0:
                                input_samples[_n].append(inp[0].detach().cpu())

                        handles.append(target_mod.register_forward_hook(_capture_awq_inp))
                    try:
                        for sample in cache_q:
                            x_q = sample["hidden_states"].to(dev)
                            kw_q = _qwen_text_kw_from_cache(sample, dev)
                            _ = _qwen_run_text_layer(layer, x_q, kw_q)
                    finally:
                        for h in handles:
                            h.remove()

                    for local_name in group_present:
                        awq_quantize_linear_module(
                            _qwen_get_module_by_name(layer, local_name),
                            input_samples.get(local_name, []),
                            wbits=args.wbits,
                            groupsize=args.groupsize,
                            zero_point=bool(getattr(args, "int_zero_point", True)),
                            module_name=local_name,
                        )
                        pbar.update(1)
                    continue
                helper, _ = _qwen_collect_module_stats_on_cached_layer(
                    layer,
                    fp_layer,
                    primary_name,
                    cache_q,
                    cache_fp,
                    _qwen_run_text_layer,
                    _qwen_text_kw_from_cache,
                    use_qep=(
                        _should_apply_qep(args, primary_name)
                        or bool(getattr(args, "gptaq", False))
                    ),
                    sequence_token_weights=(
                        getattr(args, "_qwen_sequence_token_weights", {}).get(
                            f"thinker.model.layers.{lid}"
                        )
                    ),
                )
                if helper is None:
                    pbar.update(len(group_present))
                    continue
                for local_name in group_present:
                    full_name = f"thinker.model.layers.{lid}.{local_name}"
                    alpha = float(args.perccorr)
                    if alpha_by_module is not None:
                        alpha = float(alpha_by_module.get(full_name, alpha))
                    w_q = _qwen_gptq_weight(
                        helper,
                        layer,
                        local_name,
                        full_name,
                        args,
                        alpha,
                    )
                    _qwen_get_module_by_name(layer, local_name).weight.data.copy_(
                        w_q.to(_qwen_get_module_by_name(layer, local_name).weight.data.dtype)
                    )
                    pbar.update(1)
                helper.free()
            cache_q = _qwen_propagate_cached_layer(
                layer, cache_q, _qwen_run_text_layer, _qwen_text_kw_from_cache
            )
            if need_fp and fp_layer is not None and cache_fp is not None:
                cache_fp = _qwen_propagate_cached_layer(
                    fp_layer, cache_fp, _qwen_run_text_layer, _qwen_text_kw_from_cache
                )
            if collect_scores_only and (cache_fp is not None):
                layer_mse = float(_qwen_hidden_mse_from_caches(cache_q, cache_fp))
                for local_name in layer_local_names:
                    full_name = f"thinker.model.layers.{lid}.{local_name}"
                    module_scores[full_name] = layer_mse
            thinker.model.layers[lid] = layer.cpu()
            if fp_layer is not None:
                fp_thinker.model.layers[lid] = fp_layer.cpu()

    if fp_model is not None:
        del fp_model
        torch.cuda.empty_cache()
    if pbar is not None:
        pbar.close()
    return module_scores


# ---------------------------------------------------------------------------
# Voxtral family adapter: Whisper-large-v3-style audio tower + Llama backbone.
#
# The audio tower layers are structurally whisper encoder layers (signature
# (hidden_states, attention_mask, layer_head_mask)), but calibration follows
# the Voxtral forward path: audio embeddings are projected and scattered into
# inputs_embeds at the [AUDIO] placeholder positions (audio_token_id) before
# the Llama stack runs. The text stack reuses the Qwen text-layer runner and
# kwargs builder because Voxtral's Llama layers take the identical call
# signature.
# ---------------------------------------------------------------------------

# 30 s chunk -> 3000 mel frames -> 1500 encoder positions (conv2 stride 2)
# -> grouped 4 frames per token by the 5120-wide projector reshape.
VOXTRAL_AUDIO_TOKENS_PER_CHUNK = 375
VOXTRAL_DEFAULT_AUDIO_TOKEN_ID = 24

VOXTRAL_AUDIO_GROUPS = [
    ["self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj"],
    ["self_attn.out_proj"],
    ["fc1"],
    ["fc2"],
]

VOXTRAL_TEXT_GROUPS = [
    ["self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj"],
    ["self_attn.o_proj"],
    ["mlp.gate_proj", "mlp.up_proj"],
    ["mlp.down_proj"],
]

# Voxtral's Llama decoder layers accept exactly the kwargs the Qwen text
# runner passes (position_embeddings / attention_mask / position_ids /
# past_key_values / use_cache / cache_position).
_voxtral_run_text_layer = _qwen_run_text_layer
_voxtral_text_kw_from_cache = _qwen_text_kw_from_cache


def _voxtral_get_root(model):
    """Return the VoxtralForConditionalGeneration module, or None."""
    for owner in (model, getattr(model, "model", None)):
        if (
            owner is not None
            and hasattr(owner, "audio_tower")
            and hasattr(owner, "language_model")
        ):
            return owner
    return None


def _voxtral_text_layers(root):
    language_model = getattr(root, "language_model", None)
    text_model = getattr(language_model, "model", None)
    return getattr(text_model, "layers", None)


def _voxtral_validate_audio_token_layout(input_ids, input_features, audio_token_id):
    """Check the [AUDIO] placeholder count against the mel chunk count."""
    n_audio_tokens = int((input_ids == int(audio_token_id)).sum())
    n_chunks = int(input_features.shape[0])
    expected = n_chunks * VOXTRAL_AUDIO_TOKENS_PER_CHUNK
    if n_audio_tokens != expected:
        raise RuntimeError(
            "Voxtral audio placeholder mismatch: "
            f"{n_audio_tokens} [AUDIO] tokens for {n_chunks} mel chunk(s), "
            f"expected {expected}."
        )


def _voxtral_build_teacher_forced_entry(processor, encoded, text):
    """Append transcript + EOS after [TRANSCRIBE]; mask prompt labels."""
    tokenizer = processor.tokenizer
    prompt_ids = encoded["input_ids"]
    transcript_token_ids = tokenizer.encode(str(text), add_special_tokens=False)
    if len(transcript_token_ids) == 0:
        raise ValueError("Voxtral calibration transcript tokenized to nothing.")
    eos_id = int(tokenizer.eos_token_id)
    transcript = torch.tensor(
        [list(transcript_token_ids) + [eos_id]],
        dtype=prompt_ids.dtype,
    )
    full_ids = torch.cat([prompt_ids, transcript], dim=1)
    labels = _qwen_mask_transcript_labels(prompt_ids, full_ids)
    attention_mask = encoded.get("attention_mask")
    if attention_mask is None:
        attention_mask = torch.ones_like(prompt_ids)
    attention_mask = torch.cat(
        [attention_mask, torch.ones_like(transcript)],
        dim=1,
    )
    return full_ids, attention_mask, labels


def _voxtral_make_calibration_data(
    processor,
    model_id,
    nsamples=128,
    seed=0,
    verbose=False,
    include_references=False,
    offset=0,
    include_labels=True,
):
    """Build teacher-forced Voxtral transcription calibration examples.

    Each entry holds the transcription-request prompt (ending in
    ``lang:en [TRANSCRIBE]``) with the reference transcript + EOS appended,
    ``labels`` masking the prompt positions with -100, and the 30 s-chunked
    mel ``input_features``.
    """
    if processor is None:
        raise RuntimeError("Voxtral calibration requires the VoxtralProcessor.")
    audio_token_id = int(
        getattr(processor, "audio_token_id", VOXTRAL_DEFAULT_AUDIO_TOKEN_ID)
    )

    dataset = load_local_parquet_dataset(
        "openslr/librispeech_asr",
        "clean",
        "train.100",
    )
    dataset = dataset.shuffle(
        seed=seed,
        buffer_size=max(10_000, (int(offset) + int(nsamples)) * 20),
    )

    calibs = []
    iterator = iter(dataset)
    try:
        samples = islice(iterator, int(offset), int(offset) + int(nsamples))
        for sample in tqdm.tqdm(
            samples,
            total=int(nsamples),
            desc="Loading Voxtral calibration data",
            disable=not bool(verbose),
        ):
            audio = _decode_audio_fallback(sample["audio"])
            text = str(sample["text"])
            encoded = processor.apply_transcription_request(
                language="en",
                audio=audio,
                model_id=str(model_id),
                sampling_rate=16000,
                format=["wav"],
            )
            entry = {
                key: value
                for key, value in encoded.items()
                if torch.is_tensor(value)
            }
            for key in ("input_ids", "input_features"):
                if key not in entry:
                    raise RuntimeError(
                        f"Voxtral processor output is missing {key}."
                    )
            _voxtral_validate_audio_token_layout(
                entry["input_ids"],
                entry["input_features"],
                audio_token_id,
            )
            if include_labels:
                full_ids, attention_mask, labels = (
                    _voxtral_build_teacher_forced_entry(processor, entry, text)
                )
                entry["input_ids"] = full_ids
                entry["attention_mask"] = attention_mask
                entry["labels"] = labels
            entry["__dataset_id__"] = str(sample["id"])
            if include_references:
                entry["__audio__"] = audio
                entry["__text__"] = text
            calibs.append(entry)
    finally:
        close = getattr(iterator, "close", None)
        if close is not None:
            close()
    return calibs


def _normalize_voxtral_task_weights(
    values,
    clip_max,
    min_ess_fraction=0.0,
):
    if not 0.0 <= float(min_ess_fraction) <= 1.0:
        raise ValueError("min_ess_fraction must be in [0, 1].")
    clip_min = 1e-3
    clip_max = float(clip_max)
    if not clip_min <= 1.0 <= clip_max:
        raise ValueError("Voxtral task-Fisher clip_max must be at least 1.")
    weights = values.detach().reshape(-1).float()
    if (
        weights.numel() == 0
        or not bool(torch.isfinite(weights).all())
        or bool((weights < 0).any())
        or float(weights.sum()) <= 0.0
    ):
        raise ValueError(
            "Voxtral task-Fisher weights must be non-negative, finite, "
            "non-empty, and have a positive total."
        )
    reference = weights / weights.mean().clamp_min(1e-12)
    reference = reference.clamp_min(clip_min)
    if bool(((reference >= clip_min) & (reference <= clip_max)).all()):
        weights = reference
    elif clip_max == 1.0:
        weights = torch.ones_like(reference)
    else:
        lower = math.log(clip_min / float(reference.max())) - 1.0
        upper = math.log(clip_max / float(reference.min())) + 1.0
        for _ in range(64):
            midpoint = 0.5 * (lower + upper)
            candidate = (reference * math.exp(midpoint)).clamp(
                clip_min, clip_max
            )
            if float(candidate.mean()) > 1.0:
                upper = midpoint
            else:
                lower = midpoint
        weights = (reference * math.exp(0.5 * (lower + upper))).clamp(
            clip_min, clip_max
        )
    if float(min_ess_fraction) <= 0.0:
        return weights
    variance = weights.sub(1.0).square().mean()
    if float(variance) <= 1e-12:
        return weights
    allowed_variance = 1.0 / float(min_ess_fraction) - 1.0
    shrinkage = min(
        1.0,
        float((allowed_variance / variance).clamp_min(0.0).sqrt()),
    )
    weights = 1.0 + shrinkage * (weights - 1.0)
    return weights


def _voxtral_collect_audio_task_fisher_weights(
    model,
    calibration_data,
    dev,
    clip_max,
    min_ess_fraction=0.0,
):
    """Task-loss empirical-Fisher weights for Voxtral audio-tower rows."""
    requested_device = torch.device(dev)
    if requested_device.type == "cuda":
        model = _move_model_to_cuda_if_possible(model)
    elif hasattr(model, "to"):
        moved = model.to(requested_device)
        if moved is not None:
            model = moved
    model = _set_eval_mode(model)
    root = _voxtral_get_root(model)
    if root is None:
        raise RuntimeError("Voxtral model has no audio tower.")
    layers = root.audio_tower.layers
    dtype = next(iter(_get_core_model(model).parameters())).dtype
    by_layer = {
        f"audio_tower.layers.{index}": [] for index in range(len(layers))
    }
    summaries = {
        key: {
            "minimum": float("inf"),
            "maximum": 0.0,
            "sum": 0.0,
            "count": 0,
            "cv_sum": 0.0,
            "samples": 0,
        }
        for key in by_layer
    }
    losses = []

    for sample in calibration_data:
        captured = {}
        handles = []

        def make_hook(index):
            def hook(_module, inputs):
                captured[index] = inputs[0]

            return hook

        for index, layer in enumerate(layers):
            handles.append(layer.register_forward_pre_hook(make_hook(index)))
        try:
            root.zero_grad(set_to_none=True)
            inputs = {
                key: value.to(dev)
                for key, value in sample.items()
                if torch.is_tensor(value)
            }
            with torch.enable_grad():
                output = root(
                    input_ids=inputs["input_ids"],
                    input_features=inputs["input_features"].to(dtype),
                    attention_mask=inputs.get("attention_mask"),
                    labels=inputs["labels"],
                    use_cache=False,
                )
            if any(index not in captured for index in range(len(layers))):
                raise RuntimeError(
                    "Failed to capture every Voxtral audio layer input."
                )
            gradients = torch.autograd.grad(
                output.loss,
                [captured[index] for index in range(len(layers))],
                create_graph=False,
                retain_graph=False,
                allow_unused=False,
            )
            losses.append(float(output.loss.detach()))
            for index, gradient in enumerate(gradients):
                weights = _normalize_voxtral_task_weights(
                    gradient.detach().float().square().mean(dim=-1),
                    clip_max,
                    min_ess_fraction=min_ess_fraction,
                ).cpu()
                key = f"audio_tower.layers.{index}"
                by_layer[key].append(weights)
                summary = summaries[key]
                summary["minimum"] = min(
                    summary["minimum"], float(weights.min())
                )
                summary["maximum"] = max(
                    summary["maximum"], float(weights.max())
                )
                summary["sum"] += float(weights.sum())
                summary["count"] += int(weights.numel())
                summary["cv_sum"] += float(weights.std(unbiased=False))
                summary["samples"] += 1
        finally:
            for handle in handles:
                handle.remove()
            root.zero_grad(set_to_none=True)

    for summary in summaries.values():
        summary["mean"] = summary.pop("sum") / max(summary["count"], 1)
        summary["mean_cv"] = summary.pop("cv_sum") / max(
            summary.pop("samples"), 1
        )
    torch.cuda.empty_cache()
    return by_layer, summaries, (
        sum(losses) / len(losses) if losses else None
    )


class _VoxtralCatcherAudio(nn.Module):
    def __init__(self, layer, store):
        super().__init__()
        self.layer = layer
        self.store = store

    def forward(self, hidden_states, attention_mask=None, layer_head_mask=None, **kwargs):
        self.store["hidden_states"] = hidden_states.detach()
        self.store["attention_mask"] = (
            attention_mask.detach() if torch.is_tensor(attention_mask) else attention_mask
        )
        self.store["layer_head_mask"] = (
            layer_head_mask.detach() if torch.is_tensor(layer_head_mask) else layer_head_mask
        )
        raise RuntimeError("Voxtral audio catcher stop")


class _VoxtralCatcherText(nn.Module):
    def __init__(self, layer, store):
        super().__init__()
        self.layer = layer
        self.store = store

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        use_cache=False,
        cache_position=None,
        position_embeddings=None,
        **kwargs,
    ):
        self.store["hidden_states"] = hidden_states.detach()
        self.store["position_embeddings"] = (
            None
            if position_embeddings is None
            else tuple(
                x.detach() if torch.is_tensor(x) else x
                for x in position_embeddings
            )
        )
        self.store["attention_mask"] = (
            attention_mask.detach() if torch.is_tensor(attention_mask) else attention_mask
        )
        self.store["position_ids"] = (
            position_ids.detach() if torch.is_tensor(position_ids) else position_ids
        )
        self.store["past_key_values"] = past_key_values
        self.store["use_cache"] = use_cache
        self.store["cache_position"] = (
            cache_position.detach() if torch.is_tensor(cache_position) else cache_position
        )
        raise RuntimeError("Voxtral text catcher stop")


def _voxtral_run_audio_layer(layer, hidden_states, layer_kwargs):
    out = layer(
        hidden_states,
        attention_mask=layer_kwargs.get("attention_mask", None),
        layer_head_mask=layer_kwargs.get("layer_head_mask", None),
    )
    if isinstance(out, tuple):
        out = out[0]
    return out


def _voxtral_audio_kw_from_cache(sample, dev):
    return {
        "attention_mask": None
        if sample["attention_mask"] is None
        else sample["attention_mask"].to(dev),
        "layer_head_mask": None
        if sample["layer_head_mask"] is None
        else sample["layer_head_mask"].to(dev),
    }


def _voxtral_collect_audio_layer0_cache(calibration_data, model, dev, dtype, verbose=False):
    root = _voxtral_get_root(model)
    if root is None or not hasattr(root.audio_tower, "layers"):
        return []
    layers = root.audio_tower.layers
    if len(layers) == 0:
        return []
    orig = layers[0]
    audio_cache = []

    for sample in tqdm.tqdm(
        calibration_data,
        desc="Voxtral cache audio layer0",
        disable=not bool(verbose),
    ):
        input_features = sample["input_features"].to(device=dev, dtype=dtype)
        store = {}
        layers[0] = _VoxtralCatcherAudio(orig, store)
        try:
            with torch.no_grad():
                _ = root.audio_tower(input_features)
        except RuntimeError as e:
            if "Voxtral audio catcher stop" not in str(e):
                raise
        finally:
            layers[0] = orig

        audio_cache.append(
            {
                "hidden_states": store["hidden_states"].detach().cpu(),
                "attention_mask": None
                if store["attention_mask"] is None
                else store["attention_mask"].detach().cpu(),
                "layer_head_mask": None
                if store["layer_head_mask"] is None
                else store["layer_head_mask"].detach().cpu(),
            }
        )
    return audio_cache


def _voxtral_collect_text_layer0_cache(calibration_data, model, dev, dtype, verbose=False):
    """Capture Llama layer-0 inputs through the full Voxtral forward.

    The full forward projects the (possibly already quantized) audio tower
    output and scatters it into inputs_embeds at the audio_token_id
    placeholder positions before the text stack runs.
    """
    root = _voxtral_get_root(model)
    if root is None:
        return []
    layers = _voxtral_text_layers(root)
    if layers is None or len(layers) == 0:
        return []
    orig = layers[0]
    text_cache = []

    for sample in tqdm.tqdm(
        calibration_data,
        desc="Voxtral cache text layer0",
        disable=not bool(verbose),
    ):
        inputs = {
            k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in sample.items()
        }
        inputs["input_features"] = inputs["input_features"].to(dtype)

        store = {}
        layers[0] = _VoxtralCatcherText(orig, store)
        try:
            with torch.no_grad():
                _ = root(
                    input_ids=inputs["input_ids"],
                    input_features=inputs["input_features"],
                    attention_mask=inputs.get("attention_mask", None),
                    use_cache=False,
                )
        except RuntimeError as e:
            if "Voxtral text catcher stop" not in str(e):
                raise
        finally:
            layers[0] = orig

        if store.get("position_embeddings") is None:
            raise RuntimeError(
                "Voxtral text layer 0 received no position embeddings."
            )
        text_cache.append(
            {
                "hidden_states": store["hidden_states"].detach().cpu(),
                "position_embeddings": tuple(
                    x.detach().cpu() if torch.is_tensor(x) else x
                    for x in store["position_embeddings"]
                ),
                "attention_mask": None
                if store["attention_mask"] is None
                else store["attention_mask"].detach().cpu(),
                "position_ids": None
                if store["position_ids"] is None
                else store["position_ids"].detach().cpu(),
                "past_key_values": None,
                "use_cache": False,
                "cache_position": None
                if store["cache_position"] is None
                else store["cache_position"].detach().cpu(),
            }
        )
    return text_cache


@torch.no_grad()
def _voxtral_quantize_cached_sequential(
    model,
    calibration_data,
    args,
    dev,
    alpha_by_module=None,
    collect_scores_only=False,
):
    """Layer-cached sequential quantization over both Voxtral stacks.

    Mirrors the Qwen cached-sequential mechanism: capture layer-0 inputs per
    stack, quantize group-by-group with GPTQ/AWQ statistics collected on the
    cached stream, and propagate the cache through each quantized layer.
    """
    model = _move_model_to_cuda_if_possible(model)
    model = _set_eval_mode(model)
    root = _voxtral_get_root(model)
    if root is None:
        raise RuntimeError("Voxtral model has no audio_tower/language_model.")
    dtype = next(iter(_get_core_model(model).parameters())).dtype
    quant_scope = getattr(args, "quant_scope", "full")
    if quant_scope not in {
        "full",
        "encoder",
        "text-backbone",
        "text-attention",
        "text-ffn",
    }:
        raise ValueError(
            f"Unsupported Voxtral quantization scope: {quant_scope!r}."
        )

    fp_model = None
    fp_root = None
    need_fp = bool(
        args.qep or args.dynamic_alpha or getattr(args, "gptaq", False)
    )
    if need_fp:
        fp_model = copy.deepcopy(model)
        fp_model = _move_model_to_cuda_if_possible(fp_model)
        fp_model = _set_eval_mode(fp_model)
        fp_root = _voxtral_get_root(fp_model)

    module_scores = {}
    if collect_scores_only:
        pbar_desc = "Voxtral score"
    elif args.method == "awq":
        pbar_desc = "Voxtral AWQ"
    elif args.method == "rtn":
        pbar_desc = "Voxtral RTN"
    else:
        pbar_desc = "Voxtral GPTQ"

    def _count_target_modules(layers, groups, stack):
        count = 0
        for lid in range(len(layers)):
            mods = dict(layers[lid].named_modules())
            for group in groups:
                if not _qwen_scope_includes_group(quant_scope, stack, group):
                    continue
                for local_name in group:
                    if local_name in mods and not _should_protect_ffn_output(
                        args,
                        local_name,
                        layer_idx=lid,
                        total_layers=len(layers),
                    ):
                        count += 1
        return count

    audio_layers = getattr(root.audio_tower, "layers", None)
    text_layers = _voxtral_text_layers(root)
    total_modules = 0
    if _qwen_scope_includes_stack(quant_scope, "audio") and audio_layers is not None:
        total_modules += _count_target_modules(
            audio_layers,
            VOXTRAL_AUDIO_GROUPS,
            "audio",
        )
    if _qwen_scope_includes_stack(quant_scope, "text") and text_layers is not None:
        total_modules += _count_target_modules(
            text_layers,
            VOXTRAL_TEXT_GROUPS,
            "text",
        )
    pbar = tqdm.tqdm(
        total=total_modules,
        desc=pbar_desc,
        disable=False,
        mininterval=0.5,
        dynamic_ncols=True,
        file=sys.stdout,
    )

    def _quantize_stack(
        layers,
        fp_layers,
        groups,
        stack,
        name_prefix,
        runner_fn,
        kw_builder_fn,
        cache_q,
        cache_fp,
    ):
        for lid in range(len(layers)):
            layer = layers[lid].to(dev).eval()
            fp_layer = None
            if fp_layers is not None:
                fp_layer = fp_layers[lid].to(dev).eval()
            layer_local_names = []
            for group in groups:
                if not _qwen_scope_includes_group(quant_scope, stack, group):
                    continue
                mods = dict(layer.named_modules())
                group_present = [n for n in group if n in mods]
                if len(group_present) == 0:
                    continue
                active_names = []
                for local_name in group_present:
                    if _should_protect_ffn_output(
                        args,
                        local_name,
                        layer_idx=lid,
                        total_layers=len(layers),
                    ):
                        full_name = f"{name_prefix}.{lid}.{local_name}"
                        _record_fp16_protected_module(
                            args,
                            full_name,
                            mods[local_name],
                        )
                    else:
                        active_names.append(local_name)
                group_present = active_names
                if len(group_present) == 0:
                    continue
                if not collect_scores_only:
                    _record_group_bits(
                        args,
                        [mods[name] for name in group_present],
                        args.wbits,
                    )
                layer_local_names.extend(group_present)
                primary_name = group_present[0]
                if args.method == "awq":
                    input_samples = {n: [] for n in group_present}
                    handles = []
                    for local_name in group_present:
                        target_mod = _qwen_get_module_by_name(layer, local_name)

                        def _capture_awq_inp(_, inp, __, _n=local_name):
                            if isinstance(inp, (tuple, list)) and len(inp) > 0:
                                input_samples[_n].append(inp[0].detach().cpu())

                        handles.append(
                            target_mod.register_forward_hook(_capture_awq_inp)
                        )
                    try:
                        for sample in cache_q:
                            x_q = sample["hidden_states"].to(dev)
                            kw_q = kw_builder_fn(sample, dev)
                            _ = runner_fn(layer, x_q, kw_q)
                    finally:
                        for h in handles:
                            h.remove()

                    for local_name in group_present:
                        awq_quantize_linear_module(
                            _qwen_get_module_by_name(layer, local_name),
                            input_samples.get(local_name, []),
                            wbits=args.wbits,
                            groupsize=args.groupsize,
                            zero_point=bool(getattr(args, "int_zero_point", True)),
                            module_name=local_name,
                        )
                        pbar.update(1)
                    continue
                helper, _ = _qwen_collect_module_stats_on_cached_layer(
                    layer,
                    fp_layer,
                    primary_name,
                    cache_q,
                    cache_fp,
                    runner_fn,
                    kw_builder_fn,
                    # GPTAQ needs the dual-stream cross Gram H_delta for
                    # every quantized group, like QEP.
                    use_qep=(
                        _should_apply_qep(args, primary_name)
                        or bool(getattr(args, "gptaq", False))
                    ),
                    sequence_token_weights=(
                        getattr(args, "_voxtral_audio_token_weights", {}).get(
                            f"audio_tower.layers.{lid}"
                        )
                        if stack == "audio"
                        else None
                    ),
                )
                if helper is None:
                    pbar.update(len(group_present))
                    continue
                for local_name in group_present:
                    full_name = f"{name_prefix}.{lid}.{local_name}"
                    alpha = float(args.perccorr)
                    if alpha_by_module is not None:
                        alpha = float(alpha_by_module.get(full_name, alpha))
                    w_q = _qwen_gptq_weight(
                        helper,
                        layer,
                        local_name,
                        full_name,
                        args,
                        alpha,
                    )
                    target = _qwen_get_module_by_name(layer, local_name)
                    target.weight.data.copy_(w_q.to(target.weight.data.dtype))
                    pbar.update(1)
                helper.free()
            cache_q = _qwen_propagate_cached_layer(
                layer, cache_q, runner_fn, kw_builder_fn
            )
            if need_fp and fp_layer is not None and cache_fp is not None:
                cache_fp = _qwen_propagate_cached_layer(
                    fp_layer, cache_fp, runner_fn, kw_builder_fn
                )
            if collect_scores_only and (cache_fp is not None):
                layer_mse = float(_qwen_hidden_mse_from_caches(cache_q, cache_fp))
                for local_name in layer_local_names:
                    full_name = f"{name_prefix}.{lid}.{local_name}"
                    module_scores[full_name] = layer_mse
            layers[lid] = layer.cpu()
            if fp_layers is not None:
                fp_layers[lid] = fp_layer.cpu()

    if _qwen_scope_includes_stack(quant_scope, "audio") and audio_layers is not None:
        cache_q = _voxtral_collect_audio_layer0_cache(
            calibration_data, model, dev, dtype, verbose=bool(args.verbose)
        )
        cache_fp = None
        if need_fp and fp_model is not None:
            cache_fp = _voxtral_collect_audio_layer0_cache(
                calibration_data, fp_model, dev, dtype, verbose=bool(args.verbose)
            )
        _quantize_stack(
            audio_layers,
            None if fp_root is None else fp_root.audio_tower.layers,
            VOXTRAL_AUDIO_GROUPS,
            "audio",
            "audio_tower.layers",
            _voxtral_run_audio_layer,
            _voxtral_audio_kw_from_cache,
            cache_q,
            cache_fp,
        )

    if _qwen_scope_includes_stack(quant_scope, "text") and text_layers is not None:
        model = _move_model_to_cuda_if_possible(model)
        model = _set_eval_mode(model)
        if fp_model is not None:
            fp_model = _move_model_to_cuda_if_possible(fp_model)
            fp_model = _set_eval_mode(fp_model)

        cache_q = _voxtral_collect_text_layer0_cache(
            calibration_data, model, dev, dtype, verbose=bool(args.verbose)
        )
        cache_fp = None
        if need_fp and fp_model is not None:
            cache_fp = _voxtral_collect_text_layer0_cache(
                calibration_data, fp_model, dev, dtype, verbose=bool(args.verbose)
            )
        _quantize_stack(
            text_layers,
            None if fp_root is None else _voxtral_text_layers(fp_root),
            VOXTRAL_TEXT_GROUPS,
            "text",
            "language_model.model.layers",
            _voxtral_run_text_layer,
            _voxtral_text_kw_from_cache,
            cache_q,
            cache_fp,
        )

    if fp_model is not None:
        del fp_model
        torch.cuda.empty_cache()
    pbar.close()
    return module_scores


def _supports_asr_pipeline(model):
    core = _get_core_model(model)
    if core is None:
        return False
    encoder = getattr(core, "encoder", None)
    decoder = getattr(core, "decoder", None)
    if encoder is None or decoder is None:
        return False
    return hasattr(encoder, "layers") and hasattr(decoder, "layers")


def _move_model_to_cuda_if_possible(model):
    if not torch.cuda.is_available():
        return model

    if hasattr(model, "to"):
        try:
            moved = model.to("cuda")
            if moved is not None:
                return moved
        except Exception:
            pass

    core = getattr(model, "model", None)
    if hasattr(core, "to"):
        try:
            core.to("cuda")
        except Exception:
            pass

    return model


def iter_rtn_named_modules(
    model,
    include_conv2d=False,
    include_lm_head=True,
    quant_scope="full",
):
    core = _get_core_model(model)
    if core is None:
        return

    seen = set()

    def _yield_once(name, module):
        if id(module) in seen:
            return
        seen.add(id(module))
        yield name, module

    # 1) Whisper/Moonshine-style encoder-decoder: keep RTN target set aligned with other quant modes.
    if (
        hasattr(core, "encoder")
        and hasattr(core, "decoder")
        and hasattr(core.encoder, "layers")
        and hasattr(core.decoder, "layers")
    ):
        enc_groups = [
            ["self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj"],
            ["self_attn.out_proj"],
            ["fc1"],
            ["fc2"],
        ]
        dec_groups = [
            ["self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj"],
            ["self_attn.out_proj"],
            ["encoder_attn.k_proj", "encoder_attn.v_proj", "encoder_attn.q_proj"],
            ["encoder_attn.out_proj"],
            ["fc1"],
            ["fc2"],
        ]
        if quant_scope in {"full", "encoder", "full-except-cross-attn"}:
            for lid, layer in enumerate(core.encoder.layers):
                full = dict(layer.named_modules())
                for group in enc_groups:
                    resolved = _resolve_layer_group_if_present(full, group)
                    if resolved is None:
                        continue
                    for local in resolved:
                        mod = full.get(local, None)
                        if isinstance(mod, nn.Linear):
                            yield from _yield_once(f"enc.layer{lid}.{local}", mod)
        if quant_scope != "encoder":
            for lid, layer in enumerate(core.decoder.layers):
                full = dict(layer.named_modules())
                for group in dec_groups:
                    group_name = group[0]
                    if quant_scope == "decoder-self-attn" and not group_name.startswith(
                        "self_attn."
                    ):
                        continue
                    if quant_scope == "decoder-cross-attn" and not group_name.startswith(
                        "encoder_attn."
                    ):
                        continue
                    if quant_scope == "decoder-ffn" and group_name not in {"fc1", "fc2"}:
                        continue
                    if (
                        quant_scope == "full-except-cross-attn"
                        and group_name.startswith("encoder_attn.")
                    ):
                        continue
                    resolved = _resolve_layer_group_if_present(full, group)
                    if resolved is None:
                        continue
                    for local in resolved:
                        mod = full.get(local, None)
                        if isinstance(mod, nn.Linear):
                            yield from _yield_once(f"dec.layer{lid}.{local}", mod)
        return

    # 2) Qwen-ASR thinker: keep RTN targets aligned with Qwen GPTQ/AWQ groups.
    thinker = _qwen_get_thinker(model)
    if thinker is not None:
        if quant_scope not in {
            "full",
            "encoder",
            "text-backbone",
            "text-attention",
            "text-ffn",
        }:
            raise ValueError(
                "Qwen RTN supports quantization scopes full and text-backbone."
            )
        if (
            _qwen_scope_includes_stack(quant_scope, "audio")
            and hasattr(thinker, "audio_tower")
            and hasattr(thinker.audio_tower, "layers")
        ):
            for lid, layer in enumerate(thinker.audio_tower.layers):
                full = dict(layer.named_modules())
                for group in QWEN_AUDIO_GROUPS:
                    if not _qwen_scope_includes_group(quant_scope, "audio", group):
                        continue
                    for local in group:
                        mod = full.get(local, None)
                        if isinstance(mod, nn.Linear):
                            yield from _yield_once(f"qwen_audio.layer{lid}.{local}", mod)
        if (
            _qwen_scope_includes_stack(quant_scope, "text")
            and hasattr(thinker, "model")
            and hasattr(thinker.model, "layers")
        ):
            for lid, layer in enumerate(thinker.model.layers):
                full = dict(layer.named_modules())
                for group in QWEN_TEXT_GROUPS:
                    if not _qwen_scope_includes_group(quant_scope, "text", group):
                        continue
                    for local in group:
                        mod = full.get(local, None)
                        if isinstance(mod, nn.Linear):
                            yield from _yield_once(f"qwen_text.layer{lid}.{local}", mod)
        return

    # 3) Voxtral: audio tower + Llama text backbone, aligned with the
    #    cached-sequential GPTQ/AWQ target groups.
    voxtral_root = _voxtral_get_root(model)
    if voxtral_root is not None:
        if quant_scope not in {
            "full",
            "encoder",
            "text-backbone",
            "text-attention",
            "text-ffn",
        }:
            raise ValueError(
                "Voxtral RTN supports quantization scopes full, encoder "
                "(audio tower only), text-backbone, text-attention, and "
                "text-ffn."
            )
        audio_layers = getattr(voxtral_root.audio_tower, "layers", None)
        if (
            _qwen_scope_includes_stack(quant_scope, "audio")
            and audio_layers is not None
        ):
            for lid, layer in enumerate(audio_layers):
                full = dict(layer.named_modules())
                for group in VOXTRAL_AUDIO_GROUPS:
                    if not _qwen_scope_includes_group(quant_scope, "audio", group):
                        continue
                    for local in group:
                        mod = full.get(local, None)
                        if isinstance(mod, nn.Linear):
                            yield from _yield_once(
                                f"voxtral_audio.layer{lid}.{local}", mod
                            )
        text_layers = _voxtral_text_layers(voxtral_root)
        if (
            _qwen_scope_includes_stack(quant_scope, "text")
            and text_layers is not None
        ):
            for lid, layer in enumerate(text_layers):
                full = dict(layer.named_modules())
                for group in VOXTRAL_TEXT_GROUPS:
                    if not _qwen_scope_includes_group(quant_scope, "text", group):
                        continue
                    for local in group:
                        mod = full.get(local, None)
                        if isinstance(mod, nn.Linear):
                            yield from _yield_once(
                                f"voxtral_text.layer{lid}.{local}", mod
                            )
        return

    # 4) Fallback generic scan for other model families.
    for name, module in core.named_modules():
        if isinstance(module, nn.Linear):
            if (not include_lm_head) and name.endswith("lm_head"):
                continue
            yield name, module
        elif include_conv2d and isinstance(module, nn.Conv2d):
            yield name, module


@torch.no_grad()
def rtn_quantize_module_inplace(module, wbits=4, group_size=128, zero_point=False):
    if not hasattr(module, "weight") or module.weight is None:
        return

    w = module.weight.data

    if isinstance(module, nn.Linear):
        q_w = pseudo_quantize_tensor(
            w,
            n_bit=wbits,
            q_group_size=group_size,
            zero_point=zero_point,
        )
        module.weight.data.copy_(q_w)

    elif isinstance(module, nn.Conv2d):
        orig_shape = w.shape
        w_flat = w.view(w.shape[0], -1)
        q_w_flat = pseudo_quantize_tensor(
            w_flat,
            n_bit=wbits,
            q_group_size=group_size,
            zero_point=zero_point,
        )
        module.weight.data.copy_(q_w_flat.view(orig_shape))


@torch.no_grad()
def rtn_quantize_model_inplace(
    model,
    wbits=4,
    group_size=128,
    zero_point=False,
    include_conv2d=False,
    include_lm_head=False,
    skip_keywords=None,
    quant_scope="full",
    verbose=True,
):
    if skip_keywords is None:
        skip_keywords = []

    quantized_names = []
    for name, module in iter_rtn_named_modules(
        model,
        include_conv2d=include_conv2d,
        include_lm_head=include_lm_head,
        quant_scope=quant_scope,
    ):
        if any(kw in name for kw in skip_keywords):
            if verbose:
                print(f"[skip keyword] {name}", flush=True)
            continue

        if verbose:
            print(
                f"[RTN] {name:90s} | {module.__class__.__name__:10s} | weight={tuple(module.weight.shape)}",
                flush=True,
            )
        rtn_quantize_module_inplace(
            module,
            wbits=wbits,
            group_size=group_size,
            zero_point=zero_point,
        )
        quantized_names.append(name)

    if verbose:
        print(f"Done. Quantized {len(quantized_names)} layers.", flush=True)
    return quantized_names


def _parse_skip_keywords(raw):
    if raw.strip() == "":
        return []
    return [x.strip() for x in raw.split(",") if x.strip()]


def _safe_get_state_dict(model):
    if hasattr(model, "state_dict"):
        try:
            return model.state_dict()
        except Exception:
            pass

    core = getattr(model, "model", None)
    if hasattr(core, "state_dict"):
        return core.state_dict()

    raise RuntimeError("No state_dict() available on model or model.model")
