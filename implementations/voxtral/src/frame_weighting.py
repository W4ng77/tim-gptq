"""Audio-native frame weighting for encoder calibration statistics.

Motivation: the per-example calibration loader feeds Whisper's processor one
clip at a time, and WhisperFeatureExtractor always pads to 30 s of log-mel
frames (3000). LibriSpeech calibration utterances average roughly 12 s, so
most encoder positions are synthetic silence, and the GPTQ Gram matrices for
every encoder linear (plus the decoder cross-attention K/V projections, whose
inputs are the final encoder hidden states) are dominated by padding rows.

These helpers turn the feature-extractor attention mask ("mask" mode), the
per-frame mel energy ("energy" mode), or the teacher-forced decoder
cross-attention mass ("attention" mode) into per-position row weights for
``gptq.Helper.add_batch(..., token_weights=...)``:

- mask:   padding frames get weight 0.0, speech frames 1.0.
- energy: weights proportional to linear mel power per encoder frame,
  clipped at ``clip_max`` (after normalizing to unit mean) and renormalized
  to unit mean; padding frames naturally approach zero.
- attention: soft weights proportional to how much cross-attention mass each
  encoder position receives in an FP16 teacher-forced pass (mean over
  decoder layers, heads, and decoder steps), unit-mean normalized and
  floored at ``floor`` so no row is ever zeroed.

Why the soft "attention" mode exists: binary mask weighting halved the W3
catastrophe on whisper-base (eval WER 1.03 -> 0.58) but backfired on
whisper-tiny (0.84 -> 1.35). At inference Whisper always consumes the fixed
30 s window, so the padding frames the mask removes from the calibration
statistics are genuinely present at inference time: zeroing them
manufactures a calibration/inference mismatch. The correct shape is "weight
each frame by how much the decoder actually consumes it", which is exactly
the cross-attention mass; padding keeps a small positive weight (the floor)
because the decoder does keep attending to it a little. The teacher-forced
pass runs on the not-yet-quantized model before calibration starts
(decoder_input_ids are already in the calibration data, so one forward per
sample suffices; no autoregressive generation, no model copy).

Approximation note: attention mass is defined on the *final* encoder hidden
states (the decoder consumes only those). Reusing the same per-position
vector for every encoder layer's Gram statistics is an approximation for
shallow layers, whose token mixing has not yet produced that representation.

Mask alignment: the feature mask lives on mel frames (3000). Whisper's
encoder front-end applies conv1 (kernel 3, stride 1, padding 1) then conv2
(kernel 3, stride 2, padding 1), so 3000 mel frames map to 1500 encoder
positions. For kernel 3 / padding 1 convolutions the output length equals
``len(range(0, L, stride))``, hence ``mask[..., ::stride]`` per conv stays
exactly aligned with the conv output positions.

Model-family notes:

- whisper: full support in every mode (fixed 3000 mel frames -> 1500 encoder
  positions).
- moonshine: the per-example calibration loader never pads the variable-length
  ``input_values``, so mask/energy have nothing to reweight (a true no-op,
  recorded in the run metadata). attention mode *does* apply: cross-attention
  mass is defined on whatever encoder positions exist, padded or not, so
  Moonshine rows are attention-weighted like Whisper rows.
- qwen3_asr: the audio tower uses a separate capture path that is out of
  scope; every frame-weighting mode (attention included) is recorded as a
  no-op in metadata and never applied, matching the mask-mode semantics.
"""

from __future__ import annotations

import contextlib

import torch


DEFAULT_ENERGY_CLIP_MAX = 10.0
DEFAULT_ATTENTION_FLOOR = 0.05
ZERO_WEIGHT_EPSILON = 1e-8


def permute_calibration_token_weights(weights, seed: int):
    """Permute each sample/layer independently while preserving marginals."""
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))

    def permute_one(value):
        if value is None:
            return None
        flat = value.detach().reshape(-1)
        order = torch.randperm(flat.numel(), generator=generator).to(flat.device)
        return flat[order].reshape(value.shape).clone()

    def permute_list(values):
        return [permute_one(value) for value in values]

    if isinstance(weights, dict):
        return {key: permute_list(weights[key]) for key in sorted(weights)}
    if isinstance(weights, list):
        return permute_list(weights)
    raise TypeError(
        "Task-weight permutation expects dict or list, got "
        f"{type(weights).__name__}."
    )

# First names of sequential quantization groups whose Helper statistics are
# accumulated from encoder hidden states (decoder cross-attention K/V inputs).
_CROSS_ATTN_PREFIXES = ("encoder_attn", "cross_attn")
_CROSS_ATTN_KV_LEAVES = ("k_proj", "v_proj", "key", "value")


def whisper_encoder_conv_strides(model) -> tuple[int, ...]:
    """Return the encoder front-end conv strides (mel frames -> positions)."""
    encoder = model.model.encoder
    strides = []
    for name in ("conv1", "conv2"):
        conv = getattr(encoder, name, None)
        if conv is None:
            continue
        stride = conv.stride
        strides.append(int(stride[0] if isinstance(stride, (tuple, list)) else stride))
    if not strides:
        raise TypeError("Encoder exposes no conv front-end; cannot map mel frames.")
    return tuple(strides)


def downsample_feature_mask(mask: torch.Tensor, strides) -> torch.Tensor:
    """Downsample a mel-frame mask to encoder positions via the conv strides.

    Kernel-3 / padding-1 convolutions with stride ``s`` produce
    ``floor((L - 1) / s) + 1 == len(range(0, L, s))`` outputs, so strided
    slicing keeps one mask entry per conv output position.
    """
    downsampled = mask
    for stride in strides:
        if int(stride) > 1:
            downsampled = downsampled[..., :: int(stride)]
    return downsampled


def mel_frame_energy(input_features: torch.Tensor) -> torch.Tensor:
    """Invert Whisper's log-mel normalization to linear power per mel frame.

    WhisperFeatureExtractor emits ``(log10(power) + 4) / 4`` (after dynamic
    range clamping), so ``10 ** (4 x - 4)`` recovers a quantity proportional
    to linear power; the mean over mel bins gives one energy per frame.
    """
    power = torch.pow(10.0, 4.0 * input_features.detach().float() - 4.0)
    return power.mean(dim=-2)


def average_pool_frames(values: torch.Tensor, strides) -> torch.Tensor:
    """Average-pool per-frame values down to conv output positions."""
    pooled = values
    for stride in strides:
        stride = int(stride)
        if stride <= 1:
            continue
        length = pooled.shape[-1]
        out_length = (length + stride - 1) // stride
        pad = out_length * stride - length
        if pad:
            pooled = torch.nn.functional.pad(pooled, (0, pad))
        pooled = pooled.reshape(*pooled.shape[:-1], out_length, stride).mean(dim=-1)
    return pooled


def mask_frame_weights(feature_attention_mask: torch.Tensor, strides) -> torch.Tensor:
    """Binary per-position weights: speech frames 1.0, padding frames 0.0."""
    weights = downsample_feature_mask(feature_attention_mask, strides)
    weights = weights.detach().reshape(-1).float()
    if float(weights.sum()) <= 0.0:
        raise ValueError(
            "Feature attention mask marks every frame as padding; refusing to "
            "zero the entire calibration sample."
        )
    return weights


def energy_frame_weights(
    input_features: torch.Tensor,
    strides,
    clip_max: float = DEFAULT_ENERGY_CLIP_MAX,
) -> torch.Tensor:
    """Per-position weights proportional to mel energy, unit-mean normalized."""
    energy = mel_frame_energy(input_features)
    frames = average_pool_frames(energy, strides)
    weights = frames.reshape(-1)
    weights = weights / weights.mean().clamp_min(1e-12)
    weights = weights.clamp(max=float(clip_max))
    return weights / weights.mean().clamp_min(1e-12)


def aggregate_cross_attention_mass(cross_attentions) -> torch.Tensor:
    """Average teacher-forced cross-attention over layers, heads, and steps.

    ``cross_attentions`` is the per-decoder-layer tuple emitted by a forward
    pass with ``output_attentions=True``; each entry has shape
    ``(batch, heads, decoder_steps, encoder_positions)``. The mean over every
    non-encoder axis yields one attention-mass scalar per encoder position.
    """
    if not cross_attentions or any(entry is None for entry in cross_attentions):
        raise ValueError(
            "Cross-attention weights are missing from the teacher-forced "
            "pass; the active attention implementation does not emit them."
        )
    stacked = torch.stack([entry.detach().float() for entry in cross_attentions])
    return stacked.mean(dim=tuple(range(stacked.ndim - 1)))


def attention_frame_weights(
    attention_mass: torch.Tensor,
    floor: float = DEFAULT_ATTENTION_FLOOR,
) -> torch.Tensor:
    """Unit-mean weights from per-position attention mass, floored at ``floor``.

    Normalization to unit mean happens before flooring, so the post-floor
    mean lies in ``[1, 1 + floor]``. Unlike the binary mask, padding rows
    keep a small positive weight: the decoder genuinely consumes them.
    """
    if not 0.0 <= float(floor) < 1.0:
        raise ValueError(f"Attention weight floor must be in [0, 1), got {floor!r}.")
    mass = attention_mass.detach().reshape(-1).float()
    if not bool(torch.isfinite(mass).all()) or float(mass.sum()) <= 0.0:
        raise ValueError(
            "Cross-attention mass must be finite with a positive total; "
            "refusing to weight the calibration sample."
        )
    weights = mass / mass.mean().clamp_min(1e-12)
    return weights.clamp(min=float(floor))


@contextlib.contextmanager
def _forced_eager_attention(model):
    """Temporarily force eager attention on every config reachable from model.

    SDPA (and other fused kernels) return ``None`` attention weights; only
    this teacher-forced pass needs materialized weights, so the previous
    implementation is restored afterwards.
    """
    configs = []
    seen = set()
    for module in model.modules():
        config = getattr(module, "config", None)
        if config is None or id(config) in seen:
            continue
        if hasattr(config, "_attn_implementation"):
            seen.add(id(config))
            configs.append((config, config._attn_implementation))
    for config, _previous in configs:
        config._attn_implementation = "eager"
    try:
        yield
    finally:
        for config, previous in configs:
            config._attn_implementation = previous


def _sample_encoder_input(sample: dict):
    if "input_features" in sample:
        return sample["input_features"]
    if "input_values" in sample:
        return sample["input_values"]
    return None


def _teacher_forced_cross_attentions(model, sample: dict, device, dtype):
    """One teacher-forced forward; returns the decoder cross-attention tuple.

    The encoder runs without ``output_attentions`` on purpose: encoder
    self-attention maps are (positions x positions) per layer and head and
    are never needed here. Only the decoder pass materializes weights.
    """
    inner = model.model
    encoder_input = _sample_encoder_input(sample).to(device=device, dtype=dtype)
    decoder_input_ids = sample["decoder_input_ids"].to(device=device)
    with torch.no_grad():
        encoded = inner.encoder(encoder_input)
        hidden = getattr(encoded, "last_hidden_state", None)
        if hidden is None:
            hidden = encoded[0]
        decoded = inner.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=hidden,
            output_attentions=True,
            use_cache=False,
        )
    return getattr(decoded, "cross_attentions", None)


def _cross_attentions_missing(cross_attentions) -> bool:
    return cross_attentions is None or any(
        entry is None for entry in cross_attentions
    )


def compute_attention_calibration_weights(
    model,
    calibration_data,
    floor: float = DEFAULT_ATTENTION_FLOOR,
):
    """Attention-mass frame weights for every calibration sample.

    Runs one FP16 teacher-forced forward per sample on the not-yet-quantized
    model (call this before any weight is touched), accumulates the
    cross-attention mass each final-encoder position receives, and converts
    it into unit-mean, floored row weights. Returns ``(weights, info)`` with
    the same contract as :func:`compute_calibration_frame_weights`.

    The info dict additionally reports padding-vs-speech mean weights for
    samples that carry a ``feature_attention_mask`` (Whisper): this number
    directly quantifies how the soft consumption weighting differs from the
    binary mask, which would set the padding mean to exactly zero.
    """
    if not 0.0 <= float(floor) < 1.0:
        raise ValueError(f"Attention weight floor must be in [0, 1), got {floor!r}.")
    parameter = next(iter(model.parameters()))
    device, dtype = parameter.device, parameter.dtype
    requested_impl = str(
        getattr(getattr(model, "config", None), "_attn_implementation", "unknown")
    )
    was_training = bool(getattr(model, "training", False))
    model.eval()

    weightable = [
        index
        for index, sample in enumerate(calibration_data)
        if _sample_encoder_input(sample) is not None
        and "decoder_input_ids" in sample
    ]
    eager_fallback = False
    if weightable:
        probe = _teacher_forced_cross_attentions(
            model, calibration_data[weightable[0]], device, dtype
        )
        eager_fallback = _cross_attentions_missing(probe)

    weights: list[torch.Tensor | None] = [None] * len(calibration_data)
    stats = {
        "sum": 0.0,
        "count": 0,
        "floored": 0,
        "min": float("inf"),
        "max": 0.0,
        "masked_samples": 0,
        "padding_sum": 0.0,
        "padding_count": 0,
        "padding_floored": 0,
        "speech_sum": 0.0,
        "speech_count": 0,
    }
    num_layers = None
    num_heads = None
    strides = None

    context = _forced_eager_attention(model) if eager_fallback else contextlib.nullcontext()
    with context:
        for index in weightable:
            sample = calibration_data[index]
            cross_attentions = _teacher_forced_cross_attentions(
                model, sample, device, dtype
            )
            if _cross_attentions_missing(cross_attentions):
                raise RuntimeError(
                    "Cross-attention weights are still missing after the "
                    "eager fallback; cannot compute attention frame weights."
                )
            if num_layers is None:
                num_layers = len(cross_attentions)
                num_heads = int(cross_attentions[0].shape[1])
            mass = aggregate_cross_attention_mass(cross_attentions)
            unit_mean = mass.reshape(-1) / mass.mean().clamp_min(1e-12)
            sample_weights = unit_mean.clamp(min=float(floor)).cpu()
            weights[index] = sample_weights

            values = sample_weights
            stats["sum"] += float(values.sum())
            stats["count"] += int(values.numel())
            stats["floored"] += int((unit_mean < float(floor)).sum())
            stats["min"] = min(stats["min"], float(values.min()))
            stats["max"] = max(stats["max"], float(values.max()))

            mask = sample.get("feature_attention_mask")
            if mask is None:
                continue
            if strides is None:
                strides = whisper_encoder_conv_strides(model)
            positions = (
                downsample_feature_mask(mask, strides).reshape(-1).bool().cpu()
            )
            if positions.numel() != values.numel():
                raise ValueError(
                    "Feature attention mask does not align with encoder "
                    f"positions: {positions.numel()} != {values.numel()}."
                )
            stats["masked_samples"] += 1
            padding = values[~positions]
            speech = values[positions]
            floored_padding = unit_mean.cpu()[~positions] < float(floor)
            stats["padding_sum"] += float(padding.sum())
            stats["padding_count"] += int(padding.numel())
            stats["padding_floored"] += int(floored_padding.sum())
            stats["speech_sum"] += float(speech.sum())
            stats["speech_count"] += int(speech.numel())

    if was_training:
        model.train()

    def _mean(total: float, count: int):
        return total / count if count else None

    info = {
        "mode": "attention",
        "num_samples": len(calibration_data),
        "noop_samples": len(calibration_data) - len(weightable),
        "floor": float(floor),
        "conv_strides": list(strides) if strides is not None else None,
        "attention_pass": {
            "teacher_forced": (
                "one forward per calibration sample on the not-yet-quantized "
                "model using the stored decoder_input_ids (no generation)"
            ),
            "aggregation": (
                "decoder cross-attention averaged over layers, heads, and "
                "decoder steps at the final encoder hidden states; the same "
                "vector approximates shallow encoder layers"
            ),
            "requested_attn_implementation": requested_impl,
            "eager_fallback": bool(eager_fallback),
            "num_decoder_layers": num_layers,
            "num_attention_heads": num_heads,
        },
        "weight_summary": {
            "weighted_samples": len(weightable),
            "mean_weight": _mean(stats["sum"], stats["count"]),
            "min_weight": stats["min"] if stats["count"] else None,
            "max_weight": stats["max"] if stats["count"] else None,
            "floored_fraction": (
                stats["floored"] / stats["count"] if stats["count"] else None
            ),
            "padding_vs_speech": {
                "samples_with_feature_mask": stats["masked_samples"],
                "padding_frames": stats["padding_count"],
                "speech_frames": stats["speech_count"],
                "padding_mean_weight": _mean(
                    stats["padding_sum"], stats["padding_count"]
                ),
                "speech_mean_weight": _mean(
                    stats["speech_sum"], stats["speech_count"]
                ),
                "padding_floored_fraction": (
                    stats["padding_floored"] / stats["padding_count"]
                    if stats["padding_count"]
                    else None
                ),
            },
        },
    }
    return weights, info


def compute_sample_frame_weights(
    sample: dict,
    mode: str,
    strides,
    clip_max: float = DEFAULT_ENERGY_CLIP_MAX,
) -> torch.Tensor | None:
    """Return per-encoder-position weights for one calibration sample.

    Samples without ``input_features`` (Moonshine ``input_values``,
    Qwen audio-tower rows) return ``None``: frame weighting is a no-op for
    those families, by design (see module docstring).
    """
    if mode == "none":
        return None
    if "input_features" not in sample:
        return None
    if mode == "mask":
        mask = sample.get("feature_attention_mask")
        if mask is None:
            raise ValueError(
                "mask frame weighting requires the feature attention mask; "
                "load calibration data with include_feature_attention_mask=True."
            )
        return mask_frame_weights(mask, strides)
    if mode == "energy":
        return energy_frame_weights(sample["input_features"], strides, clip_max)
    if mode == "attention":
        raise ValueError(
            "attention frame weights need a model forward pass; use "
            "compute_attention_calibration_weights (or "
            "compute_calibration_frame_weights) instead."
        )
    raise ValueError(f"Unsupported frame weighting mode: {mode!r}")


def compute_calibration_frame_weights(
    model,
    calibration_data,
    mode: str,
    clip_max: float = DEFAULT_ENERGY_CLIP_MAX,
    floor: float = DEFAULT_ATTENTION_FLOOR,
):
    """Compute per-sample frame weights aligned with the calibration order.

    Returns ``(weights, info)`` where ``weights[j]`` is a 1-D tensor over the
    encoder positions of sample ``j`` (or ``None`` for no-op samples) and
    ``info`` summarizes coverage for the run metadata. ``mode='attention'``
    runs one teacher-forced forward per sample on the (not yet quantized)
    model; call it before quantization mutates any weight.
    """
    if mode == "attention":
        return compute_attention_calibration_weights(
            model,
            calibration_data,
            floor=floor,
        )
    weights: list[torch.Tensor | None] = []
    noop_samples = 0
    strides = None
    for sample in calibration_data:
        if mode != "none" and "input_features" in sample and strides is None:
            strides = whisper_encoder_conv_strides(model)
        sample_weights = compute_sample_frame_weights(
            sample,
            mode,
            strides,
            clip_max=clip_max,
        )
        if sample_weights is None:
            noop_samples += 1
        weights.append(sample_weights)
    info = {
        "mode": mode,
        "num_samples": len(weights),
        "noop_samples": noop_samples,
        "conv_strides": list(strides) if strides is not None else None,
        "energy_clip_max": float(clip_max) if mode == "energy" else None,
    }
    return weights, info


def frame_weights_apply_to_group(layer_key_prefix: str, resolved_names) -> bool:
    """Decide whether a sequential group's Gram rows are encoder positions.

    Encoder groups always are. In the decoder, only the cross-attention K/V
    group qualifies: its Helper accumulates statistics from the first module
    (``encoder_attn.k_proj``), whose input is the final encoder hidden state.
    Decoder self-attention, cross-attention out_proj, and FFN rows are
    autoregressive token positions and stay unweighted.
    """
    if layer_key_prefix == "enc":
        return True
    if layer_key_prefix != "dec" or not resolved_names:
        return False
    first = str(resolved_names[0])
    prefix, _, leaf = first.rpartition(".")
    return prefix in _CROSS_ATTN_PREFIXES and leaf in _CROSS_ATTN_KV_LEAVES


def record_frame_weight_statistics(args, module_key: str, weights: torch.Tensor) -> None:
    """Accumulate per-module weight statistics on the args namespace."""
    stats = getattr(args, "_frame_weighting_stats", None)
    if stats is None:
        stats = {}
        args._frame_weighting_stats = stats
    entry = stats.setdefault(
        module_key,
        {"weight_sum": 0.0, "zero_rows": 0, "rows": 0},
    )
    values = weights.detach().reshape(-1).float()
    entry["weight_sum"] += float(values.sum())
    entry["zero_rows"] += int((values <= ZERO_WEIGHT_EPSILON).sum())
    entry["rows"] += int(values.numel())


def finalize_frame_weight_statistics(args) -> dict:
    """Convert accumulated statistics to mean / zero-fraction per module."""
    stats = getattr(args, "_frame_weighting_stats", None) or {}
    finalized = {}
    for module_key in sorted(stats):
        entry = stats[module_key]
        rows = max(int(entry["rows"]), 1)
        finalized[module_key] = {
            "mean_weight": entry["weight_sum"] / rows,
            "zero_weight_fraction": entry["zero_rows"] / rows,
            "rows": int(entry["rows"]),
        }
    return finalized
