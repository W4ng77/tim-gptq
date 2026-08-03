"""Randomized block-diagonal Hadamard rotations for rotation-aided GPTQ.

QuaRot/QuIP-style weight-side rotation in fake-quant form: each quantized
linear layer is quantized in a rotated input basis and folded back so the
inference path is unchanged::

    y = W x = (W R) (R^T x)          with R orthogonal
    W_hat = quantize(W R) R^T        (fold-back)

Only the weight matrix and the accumulated calibration Gram matrices need to
be conjugated (H' = R^T H R); captured activations are never re-rotated.
"""

from __future__ import annotations

import hashlib
import math

import torch


def largest_power_of_two_divisor(value: int) -> int:
    """Return the largest power of two dividing ``value`` (lowest set bit)."""
    value = int(value)
    if value <= 0:
        raise ValueError(f"Dimension must be positive, got {value}.")
    return value & (-value)


def hadamard_matrix(size: int, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    """Return the (unnormalized) Sylvester Hadamard matrix of power-of-two size."""
    size = int(size)
    if size <= 0 or (size & (size - 1)) != 0:
        raise ValueError(f"Hadamard size must be a power of two, got {size}.")
    matrix = torch.ones((1, 1), dtype=dtype)
    while matrix.shape[0] < size:
        matrix = torch.cat(
            [
                torch.cat([matrix, matrix], dim=1),
                torch.cat([matrix, -matrix], dim=1),
            ],
            dim=0,
        )
    return matrix


def module_rotation_seed(seed: int, tag: str) -> int:
    """Derive a per-module seed that is independent of layer processing order."""
    digest = hashlib.sha256(f"{int(seed)}:{tag}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**63)


def build_rotation(
    dim: int,
    seed: int,
    tag: str,
    device=None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return an orthogonal randomized block-diagonal Hadamard matrix.

    The block size is the largest power of two dividing ``dim`` (the
    broad-support fallback for non-power-of-two dims such as Moonshine's
    288/416 hidden sizes; an odd dim degrades to sign flips only). Each block
    is ``H_b diag(s) / sqrt(b)`` with Rademacher signs ``s`` drawn from a CPU
    generator seeded by ``(seed, tag)``, so the rotation is reproducible and
    does not depend on the order in which layers are quantized.
    """
    dim = int(dim)
    block = largest_power_of_two_divisor(dim)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(module_rotation_seed(seed, tag))
    signs = torch.randint(0, 2, (dim,), generator=generator, dtype=torch.int64)
    signs = (signs * 2 - 1).to(dtype=torch.float64)
    base = hadamard_matrix(block, dtype=torch.float64) / math.sqrt(block)
    rotation = torch.zeros((dim, dim), dtype=torch.float64)
    for start in range(0, dim, block):
        stop = start + block
        rotation[start:stop, start:stop] = base * signs[start:stop].unsqueeze(0)
    return rotation.to(device=device, dtype=dtype)
