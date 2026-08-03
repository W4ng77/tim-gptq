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
- propagated: Level-1 per-layer consumption sensitivity (see "Theory" below);
  every encoder layer gets its own row-weight vector obtained by
  backpropagating the attention-mass-weighted final encoder output to that
  layer's input hidden state with Hutchinson Rademacher VJP probes.
- task-fisher: the label-conditioned empirical-Fisher scalarization
  ``mean_h |d L_TF / d h_t^(l)|^2`` at every encoder layer input.  This is
  the direct task-curvature estimator; unlike attention it is not a reading
  proxy, and unlike propagated it needs no randomized representation probe.
  Its row weights are projected to unit mean with hard final lower/upper
  bounds rather than renormalized after clipping.

Theory (why "propagated" exists): the frame-weighted Hessian
``H = 2 X^T diag(w) X`` is the exact form of the second-order expansion of
the task loss under a K-FAC-style layerwise decomposition plus a
within-frame isotropy approximation; the "correct" per-frame weight for
layer ``l`` is the propagated consumption sensitivity

    g_t^(l) = E || d(consumption-weighted output) / d h_t^(l) ||^2 .

The attention mode is the zeroth-order approximation of this quantity: it
measures consumption at the *final* encoder hidden states and reuses the
same vector for every layer. That is accurate for deep layers (their output
is nearly the consumed tensor) and wrong for shallow layers, because encoder
self-attention mixes shallow padding positions into the *speech*
representations that later layers (and finally the decoder) consume: a
shallow padding frame with near-zero final attention mass still carries
consumed signal. This shallow-layer error, together with the destruction of
the attention-sink statistics, is the candidate mechanism for the binary
mask backfiring on whisper-tiny (eval WER 0.84 -> 1.35). The propagated
mode estimates g_t^(l) per layer (Level-1): construct a probe
``v = sqrt(a_t) * r`` on the final encoder hidden states (``a_t`` the
unit-mean teacher-forced cross-attention mass, ``r`` Rademacher per frame
and channel), backpropagate ``<v, h^(L)>`` to every encoder layer's input
hidden state in one VJP, and average the squared per-frame gradients over
probes and channels:

    s_t^(l) = E_r mean_h ( d<v, h^(L)> / d h^(l)_{t,h} )^2 ,

an unbiased Hutchinson estimate of ``mean_h diag(J_l^T diag(a) J_l)_{t,h}``
up to the fixed ``1/d_L`` probe scale. Unlike attention mode there is *no*
manual floor: propagated sensitivities are naturally nonzero exactly
because of the self-attention mixing above — that non-vanishing is the
principled content of the mode, not an artifact to be patched. (The
``[1e-3, 100]`` clip is numerical hygiene shared with the prophess decoder
path, not a modeling floor.) Predicted depth signature, recorded per run in
the ``flatness`` metadata: shallow layers stay flat across the
padding/speech divide (``padding_to_speech_ratio_by_layer`` near 1 — the
padding rows genuinely carry consumed signal), deep layers concentrate on
speech (ratio toward 0) and approach the peaked attention mass. The raw
per-layer CV is recorded alongside; on real whisper-tiny it *falls* with
depth because a few sink-like padding positions carry extreme shallow
sensitivity — the very rows binary masking deletes — so CV measures
spikiness, and the padding ratio is the flatness axis of the prediction.

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
  recorded in the run metadata). attention and propagated modes *do* apply:
  cross-attention mass (and the sensitivities propagated from it) is defined
  on whatever encoder positions exist, padded or not, so Moonshine rows are
  weighted like Whisper rows.
- qwen3_asr: the audio tower uses a separate cached-sequential path.
  mask/energy/attention/propagated are recorded no-ops there; task-fisher is
  implemented by the architecture adapter and weights its audio-layer rows.
"""

from __future__ import annotations

import contextlib
import math

import torch


DEFAULT_ENERGY_CLIP_MAX = 10.0
DEFAULT_ATTENTION_FLOOR = 0.05
DEFAULT_PROPAGATED_PROBES = 2
PROPAGATED_CLIP_MIN = 1e-3
PROPAGATED_CLIP_MAX = 100.0
ZERO_WEIGHT_EPSILON = 1e-8

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
    """Unit-mean weights from attention mass with a soft uniform floor.

    ``floor`` is the coefficient of an identity-metric mixture:
    ``(1-floor) * mass/mean(mass) + floor``.  This keeps every arm at exactly
    unit mean, guarantees a minimum weight of ``floor``, and matches the
    factorial terminal metric ``A=(1-floor)Diag(a/mean(a))+floor*I``.
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
    return (1.0 - float(floor)) * weights + float(floor)


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
            sample_weights = attention_frame_weights(mass, floor=floor).cpu()
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


class PropagatedFrameWeights:
    """Per-encoder-layer calibration row weights plus final-layer weights.

    ``encoder_layer_weights[l][j]`` weights the Gram rows of encoder layer
    ``l``'s sequential groups for calibration sample ``j`` (``None`` for
    no-op samples). ``cross_attention_weights[j]`` weights the decoder
    cross-attention K/V group, whose Helper consumes the final encoder
    hidden states: for that group the zeroth-order attention mass *is* the
    exact consumption sensitivity, so it needs no propagation.
    """

    mode = "propagated"

    def __init__(self, encoder_layer_weights, cross_attention_weights):
        self.encoder_layer_weights = encoder_layer_weights
        self.cross_attention_weights = cross_attention_weights

    def __len__(self):
        return len(self.cross_attention_weights)

    def weights_for_group(self, layer_key_prefix, layer_idx):
        """Per-sample weight list for one sequential quantization group."""
        if layer_key_prefix == "enc":
            index = int(layer_idx)
            if not 0 <= index < len(self.encoder_layer_weights):
                raise IndexError(
                    f"No propagated weights for encoder layer {index}; "
                    f"{len(self.encoder_layer_weights)} layers were captured."
                )
            return self.encoder_layer_weights[index]
        return self.cross_attention_weights


def permute_calibration_token_weights(weights, seed: int):
    """Deterministically destroy state alignment while preserving marginals."""
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))

    def permute_one(value):
        if value is None:
            return None
        flat = value.detach().reshape(-1)
        order = torch.randperm(flat.numel(), generator=generator)
        order = order.to(device=flat.device)
        return flat[order].reshape(value.shape).clone()

    def permute_list(values):
        return [permute_one(value) for value in values]

    if isinstance(weights, PropagatedFrameWeights):
        return PropagatedFrameWeights(
            [permute_list(layer) for layer in weights.encoder_layer_weights],
            permute_list(weights.cross_attention_weights),
        )
    if isinstance(weights, dict):
        return {
            key: permute_list(weights[key])
            for key in sorted(weights)
        }
    if isinstance(weights, list):
        return permute_list(weights)
    raise TypeError(
        "Task-weight permutation expects PropagatedFrameWeights, dict, or list; "
        f"got {type(weights).__name__}."
    )


def permute_calibration_token_weights_stratified(weights, strata_masks, seed: int):
    """Permute each sample's weights independently inside two fixed strata."""
    if not isinstance(weights, dict):
        raise TypeError(
            "Stratified task-weight permutation expects a layer-keyed dict; "
            f"got {type(weights).__name__}."
        )
    masks = list(strata_masks)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))

    def permute_layer(values):
        if len(values) != len(masks):
            raise ValueError(
                "Stratified permutation sample count mismatch: "
                f"{len(values)} weights != {len(masks)} masks."
            )
        output = []
        for sample_index, (value, mask) in enumerate(zip(values, masks)):
            flat = value.detach().reshape(-1)
            flat_mask = mask.detach().reshape(-1).bool()
            if flat.numel() != flat_mask.numel():
                raise ValueError(
                    "Stratified permutation row mismatch for sample "
                    f"{sample_index}: {flat.numel()} != {flat_mask.numel()}."
                )
            true_index = torch.nonzero(flat_mask, as_tuple=False).reshape(-1)
            false_index = torch.nonzero(~flat_mask, as_tuple=False).reshape(-1)
            if true_index.numel() == 0 or false_index.numel() == 0:
                raise ValueError(
                    "Stratified permutation requires two non-empty strata; "
                    f"sample {sample_index} has sizes "
                    f"{true_index.numel()}/{false_index.numel()}."
                )
            permuted = flat.clone()
            for index in (true_index, false_index):
                order = torch.randperm(index.numel(), generator=generator)
                source = index[order].to(device=flat.device)
                target = index.to(device=flat.device)
                permuted[target] = flat[source]
            output.append(permuted.reshape(value.shape))
        return output

    return {key: permute_layer(weights[key]) for key in sorted(weights)}


def resolve_group_frame_weights(frame_weights, layer_key_prefix, layer_idx):
    """Resolve layer-aware frame weights to one per-sample weight list.

    List-valued modes (mask/energy/attention) share one vector per sample
    across every weighted group and pass through unchanged; propagated
    weights select the target layer's vector (encoder groups) or the
    final-layer attention mass (decoder cross-attention K/V).
    """
    if isinstance(frame_weights, PropagatedFrameWeights):
        return frame_weights.weights_for_group(layer_key_prefix, layer_idx)
    return frame_weights


def normalize_propagated_weights(
    values: torch.Tensor,
    clip_min: float = PROPAGATED_CLIP_MIN,
    clip_max: float = PROPAGATED_CLIP_MAX,
) -> torch.Tensor:
    """Unit-mean normalize, clip to ``[clip_min, clip_max]``, renormalize.

    Mirrors the prophess decoder token-weight post-processing. The clip is
    numerical hygiene, not a modeling floor: propagated sensitivities are
    naturally nonzero because encoder self-attention mixes every position
    into the consumed output (see the module docstring).
    """
    weights = values.detach().reshape(-1).float()
    if not bool(torch.isfinite(weights).all()) or float(weights.sum()) <= 0.0:
        raise ValueError(
            "Propagated frame weights must be finite with a positive total; "
            "refusing to weight the calibration sample."
        )
    weights = weights / weights.mean().clamp_min(1e-12)
    weights = weights.clamp(min=float(clip_min), max=float(clip_max))
    return weights / weights.mean().clamp_min(1e-12)


def normalize_task_fisher_weights(
    values: torch.Tensor,
    clip_min: float = PROPAGATED_CLIP_MIN,
    clip_max: float = PROPAGATED_CLIP_MAX,
    min_ess_fraction: float = 0.0,
) -> torch.Tensor:
    """KL/I-project task curvature to unit mean with hard final bounds.

    After flooring the normalized reference density, the projection has the
    multiplicative water-filling form ``clip(c * reference, clip_min,
    clip_max)``, where ``c`` is chosen so the final weights have unit mean.
    This preserves density ratios among unsaturated states while guaranteeing
    the *returned* weights remain inside the requested bounds.

    Optionally mix the bounded weights with the identity metric,
    ``w_lambda = 1 + lambda (w - 1)``. Since
    ``ESS/T = 1 / (1 + lambda^2 Var(w))``, the largest admissible lambda
    satisfying the requested ESS floor is available in closed form.
    """
    if not 0.0 <= float(min_ess_fraction) <= 1.0:
        raise ValueError(
            "min_ess_fraction must be in [0, 1], got "
            f"{min_ess_fraction!r}."
        )
    clip_min = float(clip_min)
    clip_max = float(clip_max)
    if not 0.0 < clip_min <= 1.0 <= clip_max:
        raise ValueError(
            "Final bounded unit-mean weights require "
            f"0 < clip_min <= 1 <= clip_max, got [{clip_min}, {clip_max}]."
        )
    weights = values.detach().reshape(-1).float()
    if (
        weights.numel() == 0
        or not bool(torch.isfinite(weights).all())
        or bool((weights < 0).any())
        or float(weights.sum()) <= 0.0
    ):
        raise ValueError(
            "Task-Fisher weights must be non-negative, finite, non-empty, "
            "and have a positive total."
        )
    reference = weights / weights.mean().clamp_min(1e-12)
    reference = reference.clamp_min(clip_min)
    if bool(((reference >= clip_min) & (reference <= clip_max)).all()):
        bounded = reference
    elif clip_min == 1.0 or clip_max == 1.0:
        bounded = torch.ones_like(reference)
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
        bounded = (reference * math.exp(0.5 * (lower + upper))).clamp(
            clip_min, clip_max
        )
    if float(min_ess_fraction) <= 0.0:
        return bounded
    variance = bounded.sub(1.0).square().mean()
    if float(variance) <= 1e-12:
        return bounded
    allowed_variance = 1.0 / float(min_ess_fraction) - 1.0
    shrinkage = min(
        1.0,
        float((allowed_variance / variance).clamp_min(0.0).sqrt()),
    )
    return 1.0 + shrinkage * (bounded - 1.0)


def hutchinson_frame_sensitivities(sources, target, mass, probes):
    """Per-frame consumption sensitivity of ``target`` w.r.t. each source.

    For every source hidden state ``h^(l)`` (shape ``(batch, frames_l,
    hidden_l)``) this returns the Hutchinson estimate of

        s_t^(l) = E_r mean_h ( d<v, target> / d h^(l)_{t,h} )^2 ,
        v = sqrt(mass_t) * r / sqrt(d_target),  r ~ Rademacher(+-1),

    i.e. the channel-averaged diagonal of ``J_l^T diag(mass) J_l`` scaled by
    the fixed ``1/d_target`` probe factor (irrelevant after unit-mean
    normalization). One ``torch.autograd.grad`` call per probe serves every
    source at once; the graph is retained between probes and freed on the
    last one.
    """
    if int(probes) <= 0:
        raise ValueError("probes must be positive.")
    sources = list(sources)
    if not sources:
        raise ValueError("At least one propagation source is required.")
    if target.ndim != 3 or any(source.ndim != 3 for source in sources):
        raise ValueError("Propagation sources and target must be rank-3 tensors.")
    if not target.requires_grad:
        raise ValueError("Propagation target must be part of an autograd graph.")
    mass = mass.detach().reshape(-1).float()
    if mass.numel() != target.shape[-2]:
        raise ValueError(
            "Attention mass does not align with target frames: "
            f"{mass.numel()} != {target.shape[-2]}."
        )
    if not bool(torch.isfinite(mass).all()) or bool((mass < 0).any()):
        raise ValueError("Attention mass must be finite and nonnegative.")
    scale = mass.sqrt().to(device=target.device, dtype=target.dtype)
    scale = scale.reshape(1, -1, 1) * float(max(target.shape[-1], 1)) ** -0.5
    sensitivities = [
        torch.zeros(source.shape[:-1], device=source.device, dtype=torch.float32)
        for source in sources
    ]
    for probe_index in range(int(probes)):
        rademacher = torch.empty_like(target).bernoulli_(0.5)
        probe = rademacher.mul_(2.0).sub_(1.0).mul_(scale)
        gradients = torch.autograd.grad(
            outputs=target,
            inputs=sources,
            grad_outputs=probe,
            retain_graph=probe_index + 1 < int(probes),
            create_graph=False,
            allow_unused=False,
        )
        for index, gradient in enumerate(gradients):
            sensitivities[index] += gradient.detach().float().square().mean(dim=-1)
    return [entry / float(probes) for entry in sensitivities]


def _encoder_forward_with_layer_states(model, sample: dict, device, dtype):
    """Grad-enabled encoder forward capturing every layer's input hidden state.

    Returns ``(layer_inputs, final_hidden)`` where ``layer_inputs[l]`` is the
    residual-stream tensor entering encoder layer ``l`` (the shared anchor
    for all of that layer's sequential groups) and ``final_hidden`` is the
    post-layer-norm ``last_hidden_state`` the decoder consumes. The encoder
    input is marked ``requires_grad`` so the graph exists even for models
    loaded with frozen parameters.
    """
    encoder = model.model.encoder
    layers = encoder.layers
    captured: list[torch.Tensor | None] = [None] * len(layers)
    handles = []

    def make_pre_hook(index):
        def hook(_module, args):
            captured[index] = args[0]

        return hook

    for index, layer in enumerate(layers):
        handles.append(layer.register_forward_pre_hook(make_pre_hook(index)))
    encoder_input = _sample_encoder_input(sample).to(device=device, dtype=dtype)
    encoder_input = encoder_input.requires_grad_(True)
    try:
        with torch.enable_grad():
            encoded = encoder(encoder_input)
    finally:
        for handle in handles:
            handle.remove()
    hidden = getattr(encoded, "last_hidden_state", None)
    if hidden is None:
        hidden = encoded[0]
    if any(state is None for state in captured):
        raise RuntimeError(
            "Failed to capture every encoder layer's input hidden state."
        )
    return captured, hidden


def _new_propagated_layer_stats() -> dict:
    return {
        "minimum": float("inf"),
        "maximum": 0.0,
        "sum": 0.0,
        "count": 0,
        "cv_sum": 0.0,
        "cv_count": 0,
        "padding_sum": 0.0,
        "padding_count": 0,
        "speech_sum": 0.0,
        "speech_count": 0,
    }


def _update_propagated_layer_stats(stats: dict, weights: torch.Tensor, positions):
    values = weights.detach().reshape(-1).float()
    stats["minimum"] = min(stats["minimum"], float(values.min()))
    stats["maximum"] = max(stats["maximum"], float(values.max()))
    stats["sum"] += float(values.sum())
    stats["count"] += int(values.numel())
    # Weights are unit mean by construction, so std == CV per sample.
    stats["cv_sum"] += float(values.std(unbiased=False))
    stats["cv_count"] += 1
    if positions is None:
        return
    padding = values[~positions]
    speech = values[positions]
    stats["padding_sum"] += float(padding.sum())
    stats["padding_count"] += int(padding.numel())
    stats["speech_sum"] += float(speech.sum())
    stats["speech_count"] += int(speech.numel())


def _finalize_propagated_layer_stats(stats: dict) -> dict:
    def _mean(total: float, count: int):
        return total / count if count else None

    summary = {
        "minimum": stats["minimum"] if stats["count"] else None,
        "maximum": stats["maximum"] if stats["count"] else None,
        "mean_weight": _mean(stats["sum"], stats["count"]),
        "mean_cv": _mean(stats["cv_sum"], stats["cv_count"]),
    }
    if stats["padding_count"] or stats["speech_count"]:
        summary["padding_vs_speech"] = {
            "padding_frames": stats["padding_count"],
            "speech_frames": stats["speech_count"],
            "padding_mean_weight": _mean(
                stats["padding_sum"], stats["padding_count"]
            ),
            "speech_mean_weight": _mean(stats["speech_sum"], stats["speech_count"]),
        }
    return summary


def _padding_to_speech_ratio(layer_summary: dict):
    """Mean padding weight over mean speech weight (None without a mask)."""
    padding_vs_speech = layer_summary.get("padding_vs_speech")
    if not padding_vs_speech:
        return None
    padding = padding_vs_speech.get("padding_mean_weight")
    speech = padding_vs_speech.get("speech_mean_weight")
    if padding is None or speech is None or speech <= 0.0:
        return None
    return padding / speech


def compute_propagated_calibration_weights(
    model,
    calibration_data,
    probes: int = DEFAULT_PROPAGATED_PROBES,
    terminal_metric: str = "attention",
    floor: float = DEFAULT_ATTENTION_FLOOR,
    propagated_clip_max: float = PROPAGATED_CLIP_MAX,
):
    """Pull a terminal representation metric back to every encoder layer.

    ``terminal_metric='attention'`` uses the same normalized, softly floored
    decoder-attention metric as attention broadcast.  ``'uniform'`` uses the
    identity metric.  This supplies the missing factorial control:

    - uniform + broadcast: ordinary GPTQ;
    - attention + broadcast: attention weighting;
    - uniform + pullback: propagated-uniform;
    - attention + pullback: propagated.

    Returns ``(weights, info)`` where ``weights`` is a
    :class:`PropagatedFrameWeights` (encoder layer ``l`` uses ``s^(l)``, the
    decoder cross-attention K/V group uses the terminal metric itself) and
    ``info`` records per-layer weight statistics plus the depth-flatness
    metric.

    Call on the not-yet-quantized model, before any weight is touched.
    """
    if int(probes) <= 0:
        raise ValueError(f"Propagated probes must be positive, got {probes!r}.")
    if terminal_metric not in {"attention", "uniform"}:
        raise ValueError(
            "terminal_metric must be 'attention' or 'uniform', got "
            f"{terminal_metric!r}."
        )
    if not 0.0 <= float(floor) < 1.0:
        raise ValueError(f"Attention weight floor must be in [0, 1), got {floor!r}.")
    if float(propagated_clip_max) < PROPAGATED_CLIP_MIN:
        raise ValueError(
            "propagated_clip_max must be at least "
            f"{PROPAGATED_CLIP_MIN}, got {propagated_clip_max!r}."
        )
    parameter = next(iter(model.parameters()))
    device, dtype = parameter.device, parameter.dtype
    requested_impl = str(
        getattr(getattr(model, "config", None), "_attn_implementation", "unknown")
    )
    was_training = bool(getattr(model, "training", False))
    model.eval()

    num_layers = len(model.model.encoder.layers)
    weightable = [
        index
        for index, sample in enumerate(calibration_data)
        if _sample_encoder_input(sample) is not None
        and (
            terminal_metric == "uniform"
            or "decoder_input_ids" in sample
        )
    ]
    eager_fallback = False
    if weightable and terminal_metric == "attention":
        probe_attentions = _teacher_forced_cross_attentions(
            model, calibration_data[weightable[0]], device, dtype
        )
        eager_fallback = _cross_attentions_missing(probe_attentions)

    encoder_layer_weights: list[list[torch.Tensor | None]] = [
        [None] * len(calibration_data) for _ in range(num_layers)
    ]
    cross_attention_weights: list[torch.Tensor | None] = [None] * len(
        calibration_data
    )
    layer_stats = [_new_propagated_layer_stats() for _ in range(num_layers)]
    cross_stats = _new_propagated_layer_stats()
    num_decoder_layers = None
    num_heads = None
    strides = None

    context = (
        _forced_eager_attention(model) if eager_fallback else contextlib.nullcontext()
    )
    with context:
        for index in weightable:
            sample = calibration_data[index]
            layer_inputs, final_hidden = _encoder_forward_with_layer_states(
                model, sample, device, dtype
            )
            if terminal_metric == "attention":
                cross_attentions = _teacher_forced_cross_attentions(
                    model, sample, device, dtype
                )
                if _cross_attentions_missing(cross_attentions):
                    raise RuntimeError(
                        "Cross-attention weights are still missing after the "
                        "eager fallback; cannot compute propagated frame weights."
                    )
                if num_decoder_layers is None:
                    num_decoder_layers = len(cross_attentions)
                    num_heads = int(cross_attentions[0].shape[1])
                mass = aggregate_cross_attention_mass(cross_attentions)
                terminal_weights = attention_frame_weights(
                    mass,
                    floor=floor,
                )
                if final_hidden.shape[-2] != terminal_weights.numel():
                    raise ValueError(
                        "Cross-attention mass does not align with encoder "
                        "positions: "
                        f"{terminal_weights.numel()} != {final_hidden.shape[-2]}."
                    )
            else:
                terminal_weights = torch.ones(
                    final_hidden.shape[-2],
                    dtype=torch.float32,
                )
            sensitivities = hutchinson_frame_sensitivities(
                layer_inputs,
                final_hidden,
                terminal_weights,
                probes=probes,
            )
            del layer_inputs, final_hidden

            positions = None
            feature_mask = sample.get("feature_attention_mask")
            if feature_mask is not None:
                if strides is None:
                    strides = whisper_encoder_conv_strides(model)
                positions = (
                    downsample_feature_mask(feature_mask, strides)
                    .reshape(-1)
                    .bool()
                    .cpu()
                )
                if positions.numel() != terminal_weights.numel():
                    raise ValueError(
                        "Feature attention mask does not align with encoder "
                        "positions: "
                        f"{positions.numel()} != {terminal_weights.numel()}."
                    )
            for layer_index, sensitivity in enumerate(sensitivities):
                layer_weights = normalize_propagated_weights(
                    sensitivity,
                    clip_max=float(propagated_clip_max),
                ).cpu()
                encoder_layer_weights[layer_index][index] = layer_weights
                _update_propagated_layer_stats(
                    layer_stats[layer_index], layer_weights, positions
                )
            # The terminal K/V metric is identical to the corresponding
            # broadcast arm; only encoder-layer transport differs.
            sample_cross = terminal_weights.detach().cpu()
            cross_attention_weights[index] = sample_cross
            _update_propagated_layer_stats(cross_stats, sample_cross, positions)

    if was_training:
        model.train()

    layer_summaries = {
        f"enc.layer{layer_index}": _finalize_propagated_layer_stats(stats)
        for layer_index, stats in enumerate(layer_stats)
    }
    cross_summary = _finalize_propagated_layer_stats(cross_stats)
    mode_name = (
        "propagated" if terminal_metric == "attention" else "propagated-uniform"
    )
    info = {
        "mode": mode_name,
        "terminal_metric": terminal_metric,
        "terminal_floor": (
            float(floor) if terminal_metric == "attention" else None
        ),
        "num_samples": len(calibration_data),
        "noop_samples": len(calibration_data) - len(weightable),
        "probes": int(probes),
        "num_encoder_layers": num_layers,
        "conv_strides": list(strides) if strides is not None else None,
        "attention_pass": {
            "teacher_forced": (
                "one forward per calibration sample on the not-yet-quantized "
                "model using the stored decoder_input_ids (no generation)"
                if terminal_metric == "attention"
                else None
            ),
            "aggregation": (
                "decoder cross-attention averaged over layers, heads, and "
                "decoder steps at the final encoder hidden states"
                if terminal_metric == "attention"
                else "identity metric over final encoder positions"
            ),
            "requested_attn_implementation": requested_impl,
            "eager_fallback": bool(eager_fallback),
            "num_decoder_layers": num_decoder_layers,
            "num_attention_heads": num_heads,
        },
        "hutchinson": {
            "target": (
                "final encoder hidden state h^(L) (post layer norm), the "
                "tensor the decoder cross-attention consumes"
            ),
            "probe": (
                "v = sqrt(a_t) * r / sqrt(d_L); a_t is the selected "
                "unit-mean terminal metric and r is Rademacher per frame "
                "and channel"
            ),
            "sources": (
                "hidden state entering each encoder layer; one shared weight "
                "vector per layer's sequential groups"
            ),
            "estimand": (
                "s_t^(l) = E mean_h (d<v, h^(L)>/d h^(l)_{t,h})^2, the "
                "diagonal pullback of the selected terminal representation "
                "metric"
            ),
            "probes": int(probes),
        },
        "normalization": (
            "terminal metric: unit mean and identical to its broadcast arm; "
            "pulled-back layer metric: unit mean, clip to "
            f"[{PROPAGATED_CLIP_MIN:g}, {float(propagated_clip_max):g}], "
            "renormalize"
        ),
        "propagated_clip_min": float(PROPAGATED_CLIP_MIN),
        "propagated_clip_max": float(propagated_clip_max),
        "layer_weight_summaries": {
            **layer_summaries,
            "cross_attention_kv": cross_summary,
        },
        "flatness": {
            "metric": (
                "coefficient of variation of per-position weights (unit "
                "mean, so CV == std), averaged over calibration samples, "
                "per encoder layer"
            ),
            "prediction": (
                "theory (module docstring): shallow-layer sensitivity is "
                "flattened across the padding/speech divide by downstream "
                "self-attention mixing (padding_to_speech_ratio near 1), "
                "deep layers concentrate on the consumed speech frames "
                "(ratio toward 0) and approach the peaked attention mass; "
                "raw CV additionally reflects sink-like single-position "
                "spikes, so read it together with the padding ratio"
            ),
            "cv_by_layer": [
                layer_summaries[f"enc.layer{layer_index}"]["mean_cv"]
                for layer_index in range(num_layers)
            ],
            "cv_cross_attention_kv": cross_summary["mean_cv"],
            "padding_to_speech_ratio_by_layer": [
                _padding_to_speech_ratio(
                    layer_summaries[f"enc.layer{layer_index}"]
                )
                for layer_index in range(num_layers)
            ],
        },
    }
    return (
        PropagatedFrameWeights(encoder_layer_weights, cross_attention_weights),
        info,
    )


def _encoder_task_loss_with_layer_states(model, sample: dict, device, dtype):
    """Teacher-forced loss plus encoder layer inputs and consumed output."""
    encoder = model.model.encoder
    layers = encoder.layers
    captured: list[torch.Tensor | None] = [None] * len(layers)
    final_hidden: list[torch.Tensor | None] = [None]
    handles = []

    def make_pre_hook(index):
        def hook(_module, args):
            captured[index] = args[0]

        return hook

    def encoder_hook(_module, _inputs, output):
        hidden = getattr(output, "last_hidden_state", None)
        if hidden is None:
            hidden = output[0] if isinstance(output, (tuple, list)) else output
        final_hidden[0] = hidden

    for index, layer in enumerate(layers):
        handles.append(layer.register_forward_pre_hook(make_pre_hook(index)))
    handles.append(encoder.register_forward_hook(encoder_hook))
    model_inputs = {}
    for key in ("input_features", "input_values"):
        if key in sample:
            model_inputs[key] = sample[key].to(device=device, dtype=dtype)
    if not model_inputs:
        raise ValueError("Task-Fisher sample has no supported encoder input.")
    labels = sample["decoder_input_ids"].to(device=device)
    try:
        with torch.enable_grad():
            output = model(
                **model_inputs,
                labels=labels,
                use_cache=False,
            )
    finally:
        for handle in handles:
            handle.remove()
    if any(state is None for state in captured) or final_hidden[0] is None:
        raise RuntimeError(
            "Failed to capture encoder states for task-Fisher weighting."
        )
    loss = getattr(output, "loss", None)
    if loss is None:
        raise RuntimeError("Teacher-forced model pass returned no loss.")
    return loss, captured, final_hidden[0]


def compute_task_fisher_calibration_weights(
    model,
    calibration_data,
    propagated_clip_max: float = PROPAGATED_CLIP_MAX,
    min_ess_fraction: float = 0.0,
):
    """Empirical-Fisher task curvature at every encoder representation row.

    For each calibration example and encoder layer input ``h^(l)``, estimate

        g_t^(l) = mean_h |d L_TF / d h_t^(l)|^2.

    Under the same cross-frame and output-isotropy approximations used by the
    frame-weighted GPTQ derivation, this yields
    ``H_l = 2 X_l.T diag(g^(l)) X_l``.  The final-encoder gradient weights
    decoder cross-attention K/V inputs, so every protected representation is
    measured in the task-loss metric actually consumed downstream.
    """
    if float(propagated_clip_max) < PROPAGATED_CLIP_MIN:
        raise ValueError(
            "propagated_clip_max must be at least "
            f"{PROPAGATED_CLIP_MIN}, got {propagated_clip_max!r}."
        )
    if not 0.0 <= float(min_ess_fraction) <= 1.0:
        raise ValueError(
            "min_ess_fraction must be in [0, 1], got "
            f"{min_ess_fraction!r}."
        )
    parameter = next(iter(model.parameters()))
    device, dtype = parameter.device, parameter.dtype
    was_training = bool(getattr(model, "training", False))
    model.eval()

    num_layers = len(model.model.encoder.layers)
    weightable = [
        index
        for index, sample in enumerate(calibration_data)
        if _sample_encoder_input(sample) is not None
        and "decoder_input_ids" in sample
    ]
    encoder_layer_weights: list[list[torch.Tensor | None]] = [
        [None] * len(calibration_data) for _ in range(num_layers)
    ]
    cross_attention_weights: list[torch.Tensor | None] = [None] * len(
        calibration_data
    )
    layer_stats = [_new_propagated_layer_stats() for _ in range(num_layers)]
    cross_stats = _new_propagated_layer_stats()
    strides = None
    losses = []

    for index in weightable:
        sample = calibration_data[index]
        model.zero_grad(set_to_none=True)
        loss, layer_inputs, final_hidden = _encoder_task_loss_with_layer_states(
            model, sample, device, dtype
        )
        gradients = torch.autograd.grad(
            loss,
            [*layer_inputs, final_hidden],
            create_graph=False,
            retain_graph=False,
            allow_unused=False,
        )
        losses.append(float(loss.detach()))

        positions = None
        feature_mask = sample.get("feature_attention_mask")
        if feature_mask is not None:
            if strides is None:
                strides = whisper_encoder_conv_strides(model)
            positions = (
                downsample_feature_mask(feature_mask, strides)
                .reshape(-1)
                .bool()
                .cpu()
            )
        for layer_index, gradient in enumerate(gradients[:-1]):
            sensitivity = gradient.detach().float().square().mean(dim=-1)
            layer_weights = normalize_task_fisher_weights(
                sensitivity,
                clip_max=float(propagated_clip_max),
                min_ess_fraction=float(min_ess_fraction),
            ).cpu()
            if positions is not None and positions.numel() != layer_weights.numel():
                raise ValueError(
                    "Feature attention mask does not align with task-Fisher "
                    f"positions: {positions.numel()} != {layer_weights.numel()}."
                )
            encoder_layer_weights[layer_index][index] = layer_weights
            _update_propagated_layer_stats(
                layer_stats[layer_index], layer_weights, positions
            )
        final_sensitivity = (
            gradients[-1].detach().float().square().mean(dim=-1)
        )
        sample_cross = normalize_task_fisher_weights(
            final_sensitivity,
            clip_max=float(propagated_clip_max),
            min_ess_fraction=float(min_ess_fraction),
        ).cpu()
        cross_attention_weights[index] = sample_cross
        _update_propagated_layer_stats(cross_stats, sample_cross, positions)
        model.zero_grad(set_to_none=True)

    if was_training:
        model.train()
    layer_summaries = {
        f"enc.layer{layer_index}": _finalize_propagated_layer_stats(stats)
        for layer_index, stats in enumerate(layer_stats)
    }
    cross_summary = _finalize_propagated_layer_stats(cross_stats)
    info = {
        "mode": "task-fisher",
        "num_samples": len(calibration_data),
        "noop_samples": len(calibration_data) - len(weightable),
        "num_encoder_layers": num_layers,
        "conv_strides": list(strides) if strides is not None else None,
        "teacher_forcing": (
            "ground-truth transcript tokens; one differentiable forward and "
            "one joint VJP to all encoder layer inputs per calibration sample"
        ),
        "estimand": (
            "g_t^(l) = mean_h |d L_TF / d h_t^(l)|^2; label-conditioned "
            "empirical-Fisher scalarization of per-frame output curvature"
        ),
        "approximations": (
            "cross-frame curvature terms are dropped and each frame's "
            "output-side curvature is scalarized as g_t I"
        ),
        "normalization": (
            "KL/I-projection per sample and layer: ratio-preserving final "
            "weights in "
            f"[{PROPAGATED_CLIP_MIN:g}, {float(propagated_clip_max):g}] "
            "with unit mean"
        ),
        "propagated_clip_min": float(PROPAGATED_CLIP_MIN),
        "propagated_clip_max": float(propagated_clip_max),
        "minimum_ess_fraction": float(min_ess_fraction),
        "mean_teacher_forced_loss": (
            sum(losses) / len(losses) if losses else None
        ),
        "layer_weight_summaries": {
            **layer_summaries,
            "cross_attention_kv": cross_summary,
        },
    }
    return (
        PropagatedFrameWeights(encoder_layer_weights, cross_attention_weights),
        info,
    )


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
    if mode in ("attention", "propagated", "propagated-uniform", "task-fisher"):
        raise ValueError(
            f"{mode} frame weights need model forward passes; use "
            "compute_calibration_frame_weights instead."
        )
    raise ValueError(f"Unsupported frame weighting mode: {mode!r}")


def compute_calibration_frame_weights(
    model,
    calibration_data,
    mode: str,
    clip_max: float = DEFAULT_ENERGY_CLIP_MAX,
    floor: float = DEFAULT_ATTENTION_FLOOR,
    probes: int = DEFAULT_PROPAGATED_PROBES,
    propagated_clip_max: float = PROPAGATED_CLIP_MAX,
    task_fisher_min_ess_fraction: float = 0.0,
):
    """Compute per-sample frame weights aligned with the calibration order.

    Returns ``(weights, info)``. For mask/energy/attention, ``weights[j]``
    is a 1-D tensor over the encoder positions of sample ``j`` (or ``None``
    for no-op samples), shared by every weighted group. For
    ``mode='propagated'``, ``weights`` is a :class:`PropagatedFrameWeights`
    holding one vector per encoder layer per sample (resolve per group with
    :func:`resolve_group_frame_weights`). attention/propagated run forward
    (and, for propagated, backward) passes on the model; call them before
    quantization mutates any weight.
    """
    if mode == "attention":
        return compute_attention_calibration_weights(
            model,
            calibration_data,
            floor=floor,
        )
    if mode in ("propagated", "propagated-uniform"):
        return compute_propagated_calibration_weights(
            model,
            calibration_data,
            probes=probes,
            terminal_metric=(
                "attention" if mode == "propagated" else "uniform"
            ),
            floor=floor,
            propagated_clip_max=propagated_clip_max,
        )
    if mode == "task-fisher":
        return compute_task_fisher_calibration_weights(
            model,
            calibration_data,
            propagated_clip_max=propagated_clip_max,
            min_ess_fraction=task_fisher_min_ess_fraction,
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
