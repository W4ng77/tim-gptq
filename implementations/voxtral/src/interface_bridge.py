"""Calibration-only encoder-interface interventions for causal diagnosis."""

from __future__ import annotations

import torch

from quantization_utils import get_encoder_input


def _hidden_state(output):
    hidden = getattr(output, "last_hidden_state", None)
    if hidden is not None:
        return hidden
    if torch.is_tensor(output):
        return output
    if isinstance(output, (tuple, list)) and output:
        return output[0]
    if isinstance(output, dict) and "last_hidden_state" in output:
        return output["last_hidden_state"]
    raise TypeError(f"Cannot extract encoder hidden state from {type(output).__name__}.")


def _replace_hidden_state(output, hidden):
    if torch.is_tensor(output):
        return hidden
    if isinstance(output, tuple):
        return (hidden,) + tuple(output[1:])
    if isinstance(output, list):
        return [hidden] + list(output[1:])
    if isinstance(output, dict):
        output["last_hidden_state"] = hidden
        return output
    if hasattr(output, "last_hidden_state"):
        output.last_hidden_state = hidden
        return output
    raise TypeError(f"Cannot replace encoder hidden state in {type(output).__name__}.")


@torch.no_grad()
def fit_affine_encoder_bridge(
    model,
    reference_encoder,
    calibration_data,
    device,
    variance_epsilon=1e-8,
):
    """Fit per-channel ``FP ~= scale * quantized + bias`` at the interface."""
    quantized_encoder = model.model.encoder.to(device).eval()
    reference_encoder = reference_encoder.to(device).eval()
    dtype = next(iter(model.parameters())).dtype

    count = 0
    sum_q = None
    sum_fp = None
    sum_qq = None
    sum_qfp = None
    sum_ff = 0.0
    before_sse = 0.0
    try:
        for batch in calibration_data:
            features = get_encoder_input(batch).to(device=device, dtype=dtype)
            quantized = _hidden_state(quantized_encoder(features)).float()
            reference = _hidden_state(reference_encoder(features)).float()
            hidden_size = quantized.shape[-1]
            quantized = quantized.reshape(-1, hidden_size)
            reference = reference.reshape(-1, hidden_size)
            if quantized.shape != reference.shape:
                raise RuntimeError(
                    f"Bridge state mismatch: {tuple(quantized.shape)} != "
                    f"{tuple(reference.shape)}"
                )
            if sum_q is None:
                zeros = torch.zeros(hidden_size, device=device, dtype=torch.float64)
                sum_q = zeros.clone()
                sum_fp = zeros.clone()
                sum_qq = zeros.clone()
                sum_qfp = zeros.clone()
            q64 = quantized.double()
            fp64 = reference.double()
            count += q64.shape[0]
            sum_q += q64.sum(dim=0)
            sum_fp += fp64.sum(dim=0)
            sum_qq += (q64 * q64).sum(dim=0)
            sum_qfp += (q64 * fp64).sum(dim=0)
            sum_ff += fp64.square().sum().item()
            before_sse += (q64 - fp64).pow(2).sum().item()
    finally:
        quantized_encoder.cpu()
        reference_encoder.cpu()
        torch.cuda.empty_cache()

    if count == 0 or sum_q is None:
        raise RuntimeError("No encoder tokens were available to fit the interface bridge.")
    mean_q = sum_q / count
    mean_fp = sum_fp / count
    variance_q = sum_qq - count * mean_q.square()
    covariance = sum_qfp - count * mean_q * mean_fp
    stable = variance_q.abs() > float(variance_epsilon)
    scale = torch.ones_like(mean_q)
    scale[stable] = covariance[stable] / variance_q[stable]
    bias = mean_fp - scale * mean_q

    after_sse = (
        sum_qq * scale.square()
        + count * bias.square()
        - 2 * scale * sum_qfp
        + 2 * scale * bias * sum_q
        - 2 * bias * sum_fp
    ).sum().item()
    after_sse += sum_ff

    scale_cpu = scale.float().cpu()
    bias_cpu = bias.float().cpu()

    def apply_bridge(_module, _inputs, output):
        hidden = _hidden_state(output)
        corrected = hidden * scale_cpu.to(hidden.device, hidden.dtype)
        corrected = corrected + bias_cpu.to(hidden.device, hidden.dtype)
        return _replace_hidden_state(output, corrected)

    handle = model.model.encoder.register_forward_hook(apply_bridge)
    num_values = count * scale.numel()
    metadata = {
        "type": "per-channel-affine",
        "num_tokens": int(count),
        "hidden_size": int(scale.numel()),
        "mse_before": float(before_sse / max(num_values, 1)),
        "mse_after": float(after_sse / max(num_values, 1)),
        "scale_min": float(scale.min().item()),
        "scale_max": float(scale.max().item()),
        "bias_min": float(bias.min().item()),
        "bias_max": float(bias.max().item()),
        "unstable_channels": int((~stable).sum().item()),
    }
    return handle, metadata
