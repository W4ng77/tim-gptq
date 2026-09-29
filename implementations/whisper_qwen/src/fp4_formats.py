"""Numerically faithful software fake-quantization for two FP4 block formats.

MXFP4 (OCP Microscaling Formats v1.0): E2M1 elements, block size 32 along the
contracting dimension, one E8M0 (power-of-two) shared scale per block,
X = 2^(floor(log2(max|v|)) - emax_E2M1) with emax_E2M1 = 2, elements rounded to
nearest-even and saturated to +-6.

NVFP4 (NVIDIA Blackwell 4-bit format; 1-D numerical emulation, NOT native
execution): E2M1 elements, block size 16 along the contracting dimension, an
FP8 E4M3 block scale and an FP32 per-tensor scale:
    s_tensor   = amax(W) / (6 * 448)                       (FP32)
    s_block    = E4M3( amax(block) / 6 / s_tensor )        (clamped to 448)
    s_decoded  = float(s_block) * s_tensor                 (FP32)
    q          = RNE_E2M1( w / s_decoded ), saturated to +-6
    w_hat      = q * s_decoded

Both quantizers operate on a (rows, block) slice and are used inside GPTQ
exactly where the integer quantizer's ``find_params``/``quantize`` pair is used:
scales at group boundaries from the error-compensated weights, then per-column
rounding. All arithmetic is float32.
"""
from __future__ import annotations

import math

import torch

E2M1_VALUES = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
E2M1_MAX = 6.0
E2M1_EMAX = 2  # 6 = 1.5 * 2**2
E4M3_MAX = 448.0
MXFP4_BLOCK = 32
NVFP4_BLOCK = 16
# midpoints between consecutive positive codebook values
_E2M1_MIDPOINTS = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])


def round_e2m1(x: torch.Tensor) -> torch.Tensor:
    """Round to the nearest E2M1 value (round-half-to-even on the encoding),
    saturating at +-6. ``x`` is already divided by the block scale."""
    x = x.float()
    a = x.abs().clamp(max=E2M1_MAX)
    mids = _E2M1_MIDPOINTS.to(a.device)
    values = E2M1_VALUES.to(a.device)
    lower = torch.searchsorted(mids, a.reshape(-1), right=False).reshape(a.shape)
    upper = torch.searchsorted(mids, a.reshape(-1), right=True).reshape(a.shape)
    # lower == upper except at exact midpoints; at a midpoint mids[i] the two
    # candidates are values[i] and values[i+1]; the encoding with even mantissa
    # is values[i] when i is even (0, 1, 2, 4) and values[i+1] when i is odd.
    tie = lower != upper
    idx = torch.where(tie & (lower % 2 == 1), upper, lower)
    return torch.sign(x) * values[idx]


def e8m0_scale(block_amax: torch.Tensor) -> torch.Tensor:
    """OCP MX shared scale: 2^(floor(log2(amax)) - emax_elem); zero blocks -> 1."""
    amax = block_amax.float()
    exp = torch.floor(torch.log2(amax.clamp_min(torch.finfo(torch.float32).tiny))) - E2M1_EMAX
    # E8M0 exponent range is [-127, 127]; clamp like the spec's saturating cast
    exp = exp.clamp(-127.0, 127.0)
    scale = torch.exp2(exp)
    return torch.where(amax > 0, scale, torch.ones_like(scale))


def e4m3_cast(x: torch.Tensor) -> torch.Tensor:
    """Round-to-nearest-even cast to FP8 E4M3 (finite range +-448) and back."""
    return x.float().clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn).float()


def nvfp4_tensor_scale(weight: torch.Tensor) -> float:
    amax = float(weight.detach().abs().max())
    return amax / (E2M1_MAX * E4M3_MAX) if amax > 0 else 1.0


def block_scales(block: torch.Tensor, fmt: str, tensor_scale: float | None = None) -> torch.Tensor:
    """Per-row decoded scale for one (rows, block_size) weight slice."""
    amax = block.detach().float().abs().amax(dim=1)
    if fmt == "mxfp4":
        return e8m0_scale(amax)
    if fmt == "nvfp4":
        if tensor_scale is None:
            raise ValueError("NVFP4 needs the per-tensor FP32 scale.")
        local = e4m3_cast(amax / E2M1_MAX / tensor_scale)
        decoded = local * tensor_scale
        return torch.where(decoded > 0, decoded, torch.ones_like(decoded))
    raise ValueError(f"unknown FP4 format {fmt!r}")


def fake_quant_column(col: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Quantize one weight column (rows,) with per-row decoded scales and dequantize."""
    return round_e2m1(col.float() / scale) * scale


def fake_quant_block(block: torch.Tensor, fmt: str, tensor_scale: float | None = None) -> torch.Tensor:
    scale = block_scales(block, fmt, tensor_scale)
    return round_e2m1(block.float() / scale[:, None]) * scale[:, None]


def fake_quant_tensor(weight: torch.Tensor, fmt: str) -> torch.Tensor:
    """Whole-tensor RTN fake quantization (reference path, no GPTQ)."""
    block = MXFP4_BLOCK if fmt == "mxfp4" else NVFP4_BLOCK
    w = weight.detach().float()
    if w.shape[1] % block != 0:
        raise ValueError(f"contracting dim {w.shape[1]} not divisible by block {block}")
    ts = nvfp4_tensor_scale(w) if fmt == "nvfp4" else None
    out = torch.empty_like(w)
    for start in range(0, w.shape[1], block):
        out[:, start : start + block] = fake_quant_block(w[:, start : start + block], fmt, ts)
    return out


def block_size(fmt: str) -> int:
    if fmt == "mxfp4":
        return MXFP4_BLOCK
    if fmt == "nvfp4":
        return NVFP4_BLOCK
    raise ValueError(f"unknown FP4 format {fmt!r}")
