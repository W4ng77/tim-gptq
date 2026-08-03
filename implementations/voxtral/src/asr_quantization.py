import copy
import torch

from quantization_utils import map_scores_to_alpha
from quantization_utils import (
    _resolve_layer_group_if_present,
    _squeeze_batch,
    get_encoder_input,
)
from quantization_pipeline import (
    _quantize_stack_with_alpha,
    build_decoder_inps,
    build_encoder_inps,
)

QUANT_SCOPES = (
    "full",
    "encoder",
    "decoder",
    "decoder-self-attn",
    "decoder-cross-attn",
    "decoder-ffn",
    "full-except-cross-attn",
    "text-backbone",
    "text-attention",
    "text-ffn",
)


def _encoder_sequential(scope):
    if scope not in {"full", "encoder", "full-except-cross-attn"}:
        return []
    return [
        ["self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj"],
        ["self_attn.out_proj"],
        ["fc1"],
        ["fc2"],
    ]


def _decoder_sequential(scope):
    self_attention = [
        ["self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj"],
        ["self_attn.out_proj"],
    ]
    cross_attention = [
        ["encoder_attn.k_proj", "encoder_attn.v_proj", "encoder_attn.q_proj"],
        ["encoder_attn.out_proj"],
    ]
    ffn = [["fc1"], ["fc2"]]
    if scope in {"full", "decoder"}:
        return self_attention + cross_attention + ffn
    if scope == "decoder-self-attn":
        return self_attention
    if scope == "decoder-cross-attn":
        return cross_attention
    if scope == "decoder-ffn":
        return ffn
    if scope == "full-except-cross-attn":
        return self_attention + ffn
    return []


def _cross_attention_channel_importance(model):
    """Return encoder-channel weights induced by decoder cross-attention K/V."""
    importance = None
    count = 0
    for layer in model.model.decoder.layers:
        modules = dict(layer.named_modules())
        for name in ("encoder_attn.k_proj", "encoder_attn.v_proj"):
            module = modules.get(name)
            if module is None:
                alias = name.replace("encoder_attn.", "cross_attn.")
                module = modules.get(alias)
            if module is None or not hasattr(module, "weight"):
                continue
            weight = module.weight.detach().float()
            channel = weight.square().mean(dim=0).cpu()
            importance = channel if importance is None else importance + channel
            count += 1
    if importance is None or count == 0:
        return None
    importance = importance / count
    return importance / importance.mean().clamp_min(1e-12)


@torch.no_grad()
def collect_fp_encoder_outputs(model, dev, calibration_data):
    """Collect calibration encoder states without changing encoder weights."""
    encoder = model.model.encoder.to(dev)
    dtype = next(iter(model.parameters())).dtype
    outputs = []
    try:
        for batch in calibration_data:
            model_input = get_encoder_input(batch).to(device=dev, dtype=dtype)
            encoded = encoder(model_input)
            hidden = getattr(encoded, "last_hidden_state", None)
            if hidden is None:
                hidden = encoded[0]
            outputs.append(_squeeze_batch(hidden.detach().cpu()))
    finally:
        encoder.cpu()
        torch.cuda.empty_cache()
    return outputs


def collect_decoder_sequence_token_weights(model, dev, calibration_data):
    """Collect per-token empirical-Fisher weights for decoder FFN outputs."""
    decoder_layers = model.model.decoder.layers
    targets = []
    for index, layer in enumerate(decoder_layers):
        named = dict(layer.named_modules())
        resolved = _resolve_layer_group_if_present(named, ["fc2"])
        if resolved is None:
            raise TypeError(
                f"Decoder layer {index} has no recognized FFN output projection."
            )
        module_name = resolved[0]
        module = named[module_name]
        if not isinstance(module, torch.nn.Linear):
            raise TypeError(
                f"Decoder layer {index} FFN output is not linear: {module_name}."
            )
        targets.append((module_name, module))

    model.to(dev)
    model.eval()
    dtype = next(iter(model.parameters())).dtype
    by_layer = {
        f"dec.layer{index}.{module_name}": []
        for index, (module_name, _module) in enumerate(targets)
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

        for index, (_module_name, module) in enumerate(targets):
            handles.append(module.register_forward_hook(make_hook(index)))
        try:
            model.zero_grad(set_to_none=True)
            model_inputs = {}
            for key in ("input_features", "input_values"):
                if key in sample:
                    model_inputs[key] = sample[key].to(device=dev, dtype=dtype)
            labels = sample["decoder_input_ids"].to(dev)
            with torch.enable_grad():
                output = model(**model_inputs, labels=labels)
                output.loss.backward()
            for index, (module_name, _module) in enumerate(targets):
                if index not in captured or captured[index].grad is None:
                    raise RuntimeError(
                        f"Missing sequence gradient for decoder layer {index}."
                    )
                gradient = captured[index].grad.detach().float()
                weights = gradient.square().mean(dim=-1).reshape(-1)
                weights = weights / weights.mean().clamp_min(1e-12)
                weights = weights.clamp(min=1e-4, max=100.0).cpu()
                weights = weights / weights.mean().clamp_min(1e-12)
                key = f"dec.layer{index}.{module_name}"
                by_layer[key].append(weights)
                summary = summaries[key]
                summary["minimum"] = min(summary["minimum"], float(weights.min()))
                summary["maximum"] = max(summary["maximum"], float(weights.max()))
                summary["sum"] += float(weights.sum())
                summary["count"] += int(weights.numel())
        finally:
            for handle in handles:
                handle.remove()
            model.zero_grad(set_to_none=True)

    for summary in summaries.values():
        summary["mean"] = summary.pop("sum") / max(summary["count"], 1)
    model.cpu()
    torch.cuda.empty_cache()
    return by_layer, summaries


def _hutchinson_token_sensitivity(
    source: torch.Tensor,
    target: torch.Tensor,
    probes: int,
) -> torch.Tensor:
    """Estimate diag(J^T J) per source token with Rademacher VJPs."""
    if int(probes) <= 0:
        raise ValueError("probes must be positive.")
    if source.ndim != 3 or target.ndim != 3:
        raise ValueError("Propagation source and target must be rank-3 tensors.")
    sensitivity = torch.zeros(
        source.shape[:-1],
        device=source.device,
        dtype=torch.float32,
    )
    target_scale = float(max(target.shape[-1], 1)) ** -0.5
    for _ in range(int(probes)):
        probe = torch.empty_like(target).bernoulli_(0.5)
        probe = (probe.mul_(2.0).sub_(1.0)) * target_scale
        gradient = torch.autograd.grad(
            outputs=target,
            inputs=source,
            grad_outputs=probe,
            retain_graph=True,
            create_graph=False,
            allow_unused=False,
        )[0]
        sensitivity += gradient.detach().float().square().mean(dim=-1)
    return sensitivity / float(probes)


def collect_decoder_propagation_token_weights(
    model,
    dev,
    calibration_data,
    *,
    probes: int = 2,
):
    """Collect one-block look-ahead reconstruction curvature for decoder FFNs."""
    decoder_layers = model.model.decoder.layers
    targets = []
    for index, layer in enumerate(decoder_layers):
        named = dict(layer.named_modules())
        resolved = _resolve_layer_group_if_present(named, ["fc2"])
        if resolved is None:
            raise TypeError(
                f"Decoder layer {index} has no recognized FFN output projection."
            )
        module_name = resolved[0]
        module = named[module_name]
        if not isinstance(module, torch.nn.Linear):
            raise TypeError(
                f"Decoder layer {index} FFN output is not linear: {module_name}."
            )
        targets.append((module_name, module))

    model.to(dev)
    model.eval()
    dtype = next(iter(model.parameters())).dtype
    by_layer = {
        f"dec.layer{index}.{module_name}": []
        for index, (module_name, _module) in enumerate(targets)
    }
    summaries = {
        key: {"minimum": float("inf"), "maximum": 0.0, "sum": 0.0, "count": 0}
        for key in by_layer
    }

    for sample in calibration_data:
        ffn_outputs = {}
        layer_outputs = {}
        handles = []

        def make_ffn_hook(index):
            def hook(_module, _inputs, output):
                ffn_outputs[index] = output

            return hook

        def make_layer_hook(index):
            def hook(_module, _inputs, output):
                layer_outputs[index] = output[0] if isinstance(output, tuple) else output

            return hook

        for index, (_module_name, module) in enumerate(targets):
            handles.append(module.register_forward_hook(make_ffn_hook(index)))
            handles.append(
                decoder_layers[index].register_forward_hook(make_layer_hook(index))
            )
        try:
            model.zero_grad(set_to_none=True)
            model_inputs = {}
            for key in ("input_features", "input_values"):
                if key in sample:
                    model_inputs[key] = sample[key].to(device=dev, dtype=dtype)
            labels = sample["decoder_input_ids"].to(dev)
            with torch.enable_grad():
                output = model(
                    **model_inputs,
                    labels=labels,
                    output_hidden_states=True,
                    use_cache=False,
                )
            final_hidden = output.decoder_hidden_states[-1]
            for index, (module_name, _module) in enumerate(targets):
                if index not in ffn_outputs:
                    raise RuntimeError(
                        f"Missing FFN output for decoder layer {index}."
                    )
                if index + 1 < len(targets):
                    lookahead = layer_outputs[index + 1]
                else:
                    lookahead = final_hidden
                weights = _hutchinson_token_sensitivity(
                    ffn_outputs[index],
                    lookahead,
                    probes=probes,
                ).reshape(-1)
                weights = weights / weights.mean().clamp_min(1e-12)
                weights = weights.clamp(min=1e-4, max=100.0).cpu()
                weights = weights / weights.mean().clamp_min(1e-12)
                key = f"dec.layer{index}.{module_name}"
                by_layer[key].append(weights)
                summary = summaries[key]
                summary["minimum"] = min(summary["minimum"], float(weights.min()))
                summary["maximum"] = max(summary["maximum"], float(weights.max()))
                summary["sum"] += float(weights.sum())
                summary["count"] += int(weights.numel())
        finally:
            for handle in handles:
                handle.remove()
            model.zero_grad(set_to_none=True)

    for summary in summaries.values():
        summary["mean"] = summary.pop("sum") / max(summary["count"], 1)
    model.cpu()
    torch.cuda.empty_cache()
    return by_layer, summaries


@torch.no_grad()
def quantize_encoder_with_alpha(
    model,
    dev,
    args,
    calibration_data,
    gptq_nsamples=None,
    collect_layer_outputs=False,
    output_sample_idx=0,
    collect_layer_mse=False,
    alpha_by_layer=None,
    plot_stats=False,
):
    use_cache = model.config.use_cache
    model.config.use_cache = False
    dtype = next(iter(model.parameters())).dtype

    inps, layer_kwargs_list, lengths = build_encoder_inps(model, dev, calibration_data)
    layers = model.model.encoder.layers
    sequential = _encoder_sequential(getattr(args, "quant_scope", "full"))
    if not sequential:
        raise ValueError(f"Quantization scope {args.quant_scope!r} excludes the encoder.")

    def run_encoder_layer(layer, x, sample_kwargs, _sample_idx):
        kwargs = dict(sample_kwargs)
        attention_mask = kwargs.pop("attention_mask", None)
        return layer(x, attention_mask, **kwargs)

    try:
        error_channel_weight = (
            _cross_attention_channel_importance(model)
            if collect_layer_mse
            else None
        )
        return _quantize_stack_with_alpha(
            layers=layers,
            inps=inps,
            layer_kwargs_list=layer_kwargs_list,
            lengths=lengths,
            args=args,
            dev=dev,
            dtype=dtype,
            sequential=sequential,
            layer_key_prefix="enc",
            run_layer=run_encoder_layer,
            gptq_nsamples=gptq_nsamples,
            collect_layer_outputs=collect_layer_outputs,
            output_sample_idx=output_sample_idx,
            collect_layer_mse=collect_layer_mse,
            error_channel_weight=error_channel_weight,
            alpha_by_layer=alpha_by_layer,
            plot_stats=plot_stats,
        )
    finally:
        model.config.use_cache = use_cache


@torch.no_grad()
def quantize_decoder_with_alpha(
    model,
    dev,
    args,
    calibration_data,
    encoder_outputs,
    gptq_nsamples=None,
    collect_layer_outputs=False,
    output_sample_idx=0,
    collect_layer_mse=False,
    alpha_by_layer=None,
    plot_stats=False,
):
    use_cache = model.config.use_cache
    model.config.use_cache = False
    dtype = next(iter(model.parameters())).dtype

    inps, layer_kwargs_list, lengths = build_decoder_inps(
        model, dev, calibration_data, encoder_outputs
    )
    layers = model.model.decoder.layers
    sequential = _decoder_sequential(getattr(args, "quant_scope", "full"))
    if not sequential:
        raise ValueError(f"Quantization scope {args.quant_scope!r} excludes the decoder.")

    def run_decoder_layer(layer, x, sample_kwargs, sample_idx):
        kwargs = dict(sample_kwargs)
        kwargs["encoder_hidden_states"] = encoder_outputs[sample_idx].unsqueeze(0).to(dev)
        return layer(x, **kwargs)

    try:
        return _quantize_stack_with_alpha(
            layers=layers,
            inps=inps,
            layer_kwargs_list=layer_kwargs_list,
            lengths=lengths,
            args=args,
            dev=dev,
            dtype=dtype,
            sequential=sequential,
            layer_key_prefix="dec",
            run_layer=run_decoder_layer,
            gptq_nsamples=gptq_nsamples,
            collect_layer_outputs=collect_layer_outputs,
            output_sample_idx=output_sample_idx,
            collect_layer_mse=collect_layer_mse,
            alpha_by_layer=alpha_by_layer,
            plot_stats=plot_stats,
        )
    finally:
        model.config.use_cache = use_cache


@torch.no_grad()
def gptq_full_calib_collect_hidden_mse_and_alpha(
    model,
    dev,
    args,
    calibration_data,
    alpha_min=0.1,
    alpha_max=0.8,
    plot_stats=False,
):
    if args.method != "gptq":
        raise ValueError("Set args.method='gptq' (or mode gptq/gptq+qep) before calling.")

    enc_out, enc_mse = quantize_encoder_with_alpha(
        model,
        dev,
        args,
        calibration_data,
        gptq_nsamples=None,
        collect_layer_mse=True,
        plot_stats=plot_stats,
    )
    _, dec_mse = quantize_decoder_with_alpha(
        model,
        dev,
        args,
        calibration_data,
        enc_out,
        gptq_nsamples=None,
        collect_layer_mse=True,
        plot_stats=plot_stats,
    )

    enc_scores = [float(x["mse"]) for x in enc_mse]
    dec_scores = [float(x["mse"]) for x in dec_mse]

    fallback_alpha = float((alpha_min + alpha_max) / 2.0)
    enc_alphas = map_scores_to_alpha(
        enc_scores,
        alpha_min=alpha_min,
        alpha_max=alpha_max,
        fallback_alpha=fallback_alpha,
        log_scale=True,
        invert=False,
    )
    dec_alphas = map_scores_to_alpha(
        dec_scores,
        alpha_min=alpha_min,
        alpha_max=alpha_max,
        fallback_alpha=fallback_alpha,
        log_scale=True,
        invert=False,
    )

    alpha_by_layer = {}
    layer_scores = {}
    for x, a in zip(enc_mse, enc_alphas):
        k = f"enc.{x['layer_idx']}"
        alpha_by_layer[k] = float(a)
        layer_scores[k] = float(x["mse"])
    for x, a in zip(dec_mse, dec_alphas):
        k = f"dec.{x['layer_idx']}"
        alpha_by_layer[k] = float(a)
        layer_scores[k] = float(x["mse"])

    return {
        "encoder_mse": enc_mse,
        "decoder_mse": dec_mse,
        "layer_scores": layer_scores,
        "alpha_by_layer": alpha_by_layer,
    }


@torch.no_grad()
def quantize_with_dynamic_alpha(
    model,
    dev,
    args,
    calibration_data,
    alpha_min=0.1,
    alpha_max=0.8,
    plot_stats=False,
):
    score_model = copy.deepcopy(model)
    score_args = copy.deepcopy(args)
    # Stage 1 (alpha scoring) should stay on plain GPTQ path.
    # Keep FP outputs as references for scoring, but disable QEP correction.
    score_args.qep = False
    score_pack = gptq_full_calib_collect_hidden_mse_and_alpha(
        score_model,
        dev,
        score_args,
        calibration_data,
        alpha_min=alpha_min,
        alpha_max=alpha_max,
        plot_stats=plot_stats,
    )
    del score_model
    torch.cuda.empty_cache()

    alpha_by_layer = score_pack["alpha_by_layer"]
    enc_out = quantize_encoder_with_alpha(
        model,
        dev,
        args,
        calibration_data,
        alpha_by_layer=alpha_by_layer,
        plot_stats=plot_stats,
    )
    quantize_decoder_with_alpha(
        model,
        dev,
        args,
        calibration_data,
        enc_out,
        alpha_by_layer=alpha_by_layer,
        plot_stats=plot_stats,
    )
    return score_pack


@torch.no_grad()
def quantize_decoder_with_dynamic_alpha(
    model,
    dev,
    args,
    calibration_data,
    alpha_min=0.1,
    alpha_max=0.8,
    plot_stats=False,
):
    """Run TIM-GPTQ legacy scoring and correction on the decoder while retaining an FP encoder."""
    score_model = copy.deepcopy(model)
    score_args = copy.deepcopy(args)
    score_args.qep = False
    score_encoder_outputs = collect_fp_encoder_outputs(
        score_model,
        dev,
        calibration_data,
    )
    _, decoder_mse = quantize_decoder_with_alpha(
        score_model,
        dev,
        score_args,
        calibration_data,
        score_encoder_outputs,
        collect_layer_mse=True,
        plot_stats=plot_stats,
    )
    del score_model
    torch.cuda.empty_cache()

    scores = [float(item["mse"]) for item in decoder_mse]
    fallback_alpha = float((alpha_min + alpha_max) / 2.0)
    alphas = map_scores_to_alpha(
        scores,
        alpha_min=alpha_min,
        alpha_max=alpha_max,
        fallback_alpha=fallback_alpha,
        log_scale=True,
        invert=False,
    )
    alpha_by_layer = {
        f"dec.{item['layer_idx']}": float(alpha)
        for item, alpha in zip(decoder_mse, alphas)
    }
    layer_scores = {
        f"dec.{item['layer_idx']}": float(item["mse"])
        for item in decoder_mse
    }

    encoder_outputs = collect_fp_encoder_outputs(
        model,
        dev,
        calibration_data,
    )
    quantize_decoder_with_alpha(
        model,
        dev,
        args,
        calibration_data,
        encoder_outputs,
        alpha_by_layer=alpha_by_layer,
        plot_stats=plot_stats,
    )
    return {
        "decoder_mse": decoder_mse,
        "layer_scores": layer_scores,
        "alpha_by_layer": alpha_by_layer,
    }
