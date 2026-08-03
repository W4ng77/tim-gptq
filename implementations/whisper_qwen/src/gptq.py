"""GPTQ statistics, quantization, and QEP weight correction."""

from __future__ import annotations

import gc
import math

import torch
import torch.nn as nn
import transformers

from quant import Quantizer, quantize
from rotation import build_rotation


torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def _damped_cholesky(
    hessian: torch.Tensor,
    percdamp: float,
    max_retries: int = 8,
    damp_multiplier: float = 10.0,
) -> tuple[torch.Tensor, float]:
    """Return a Cholesky factor after adaptive diagonal damping."""
    diagonal_mean = torch.diagonal(hessian).abs().mean().clamp_min(1e-12)
    base_damp = max(float(percdamp), 1e-8) * float(diagonal_mean)
    diagonal = torch.arange(hessian.shape[0], device=hessian.device)
    last_error = None

    for retry in range(max_retries):
        damp = base_damp * (damp_multiplier**retry)
        damped = hessian.clone()
        damped[diagonal, diagonal] += damp
        try:
            return torch.linalg.cholesky(damped), damp
        except RuntimeError as error:
            if "cholesky" not in str(error).lower():
                raise
            last_error = error

    raise RuntimeError(
        f"Cholesky failed after {max_retries} damping attempts; "
        f"last error: {last_error}"
    )


def gptq_inverse_factor(
    hessian: torch.Tensor,
    percdamp: float,
) -> tuple[torch.Tensor, float]:
    """Return the upper Cholesky factor of the damped Hessian inverse."""
    factor, damp = _damped_cholesky(hessian, percdamp)
    inverse = torch.cholesky_inverse(factor)
    return torch.linalg.cholesky(inverse, upper=True), damp


def damped_hessian_inverse(
    hessian: torch.Tensor,
    percdamp: float,
) -> tuple[torch.Tensor, float]:
    """Return the complete damped Hessian inverse used by QEP."""
    factor, damp = _damped_cholesky(hessian, percdamp)
    return torch.cholesky_inverse(factor), damp


class Helper:
    """Accumulate calibration statistics for one linear or convolutional layer."""

    def __init__(self, layer):
        self.layer = layer
        self.device = layer.weight.device
        columns = layer.weight.shape[1]
        if isinstance(layer, transformers.Conv1D):
            columns = layer.weight.shape[0]
        self.H_q = torch.zeros((columns, columns), device=self.device)
        self.H_delta = torch.zeros((columns, columns), device=self.device)
        self.nsamples = 0

    @staticmethod
    def _matrix_input(layer, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim == 2:
            inputs = inputs.unsqueeze(0)
        if isinstance(layer, (nn.Linear, transformers.Conv1D)):
            if inputs.ndim == 3:
                inputs = inputs.reshape(-1, inputs.shape[-1])
            return inputs.t()
        if isinstance(layer, nn.Conv2d):
            unfold = nn.Unfold(
                layer.kernel_size,
                dilation=layer.dilation,
                padding=layer.padding,
                stride=layer.stride,
            )
            return unfold(inputs).permute(1, 0, 2).flatten(1)
        raise TypeError(f"Unsupported layer type: {type(layer).__name__}")

    def add_batch(
        self,
        inputs: torch.Tensor,
        token_weights: torch.Tensor | None = None,
    ) -> None:
        batch_count = inputs.shape[0] if inputs.ndim > 1 else 1
        matrix = self._matrix_input(self.layer, inputs)
        if token_weights is not None:
            weights = token_weights.detach().reshape(-1).to(
                device=matrix.device,
                dtype=torch.float32,
            )
            if weights.numel() != matrix.shape[1]:
                raise ValueError(
                    "Sequence token weights do not match linear input rows: "
                    f"{weights.numel()} != {matrix.shape[1]}"
                )
            if not torch.isfinite(weights).all() or (weights < 0).any():
                raise ValueError("Sequence token weights must be finite and nonnegative.")
            matrix = matrix.float() * weights.sqrt().unsqueeze(0)
        old_count = self.nsamples
        self.nsamples += batch_count
        self.H_q *= old_count / self.nsamples
        scaled = math.sqrt(2 / self.nsamples) * matrix.float()
        self.H_q += scaled @ scaled.t()

    def add_batch_qep(
        self,
        quantized_inputs: torch.Tensor,
        reference_inputs: torch.Tensor,
    ) -> None:
        batch_count = quantized_inputs.shape[0] if quantized_inputs.ndim > 1 else 1
        quantized = self._matrix_input(self.layer, quantized_inputs)
        delta = self._matrix_input(
            self.layer,
            reference_inputs - quantized_inputs,
        )
        old_count = self.nsamples
        self.nsamples += batch_count
        scale = old_count / self.nsamples
        self.H_q *= scale
        self.H_delta *= scale
        moment_scale = math.sqrt(2 / self.nsamples)
        quantized = moment_scale * quantized.float()
        delta = moment_scale * delta.float()
        self.H_q += quantized @ quantized.t()
        self.H_delta += delta @ quantized.t()

    def run_gptq(
        self,
        layer,
        blocksize: int = 128,
        percdamp: float = 0.01,
        wbits: int = 16,
        groupsize: int = -1,
        actorder: bool = False,
        return_W: bool = False,
        gptaq_alpha: float | None = None,
        rotate: str = "none",
        rotation_seed: int = 0,
        rotation_tag: str = "",
    ):
        quantizer = Quantizer()
        quantizer.configure(wbits, perchannel=True, sym=False, mse=False)

        weights = layer.weight.detach().clone()
        if isinstance(layer, nn.Conv2d):
            weights = weights.flatten(1)
        if isinstance(layer, transformers.Conv1D):
            weights = weights.t()
        weights = weights.float()
        hessian = self.H_q.clone()
        cross_hessian = None
        if gptaq_alpha is not None:
            # GPTAQ / GPTQv2 asymmetric calibration (arXiv:2504.02692):
            # H_delta already accumulates dX X_q^T = (X_fp - X_q) X_q^T with
            # the same sqrt(2/nsamples) scaling as the official dXXT.
            cross_hessian = self.H_delta.clone()

        rotation = None
        if rotate not in ("none", "hadamard"):
            raise ValueError(f"Unsupported rotation scheme: {rotate!r}.")
        if rotate == "hadamard":
            if not isinstance(layer, nn.Linear):
                raise TypeError(
                    "Hadamard rotation supports nn.Linear layers only; got "
                    f"{type(layer).__name__}."
                )
            # Quantize in the rotated basis W' = W R with Gram matrices
            # conjugated to that basis, then fold back after quantization so
            # the inference path is unchanged: W_hat = quantize(W R) R^T.
            rotation = build_rotation(
                weights.shape[1],
                rotation_seed,
                rotation_tag,
                device=weights.device,
                dtype=weights.dtype,
            )
            weights = weights @ rotation
            hessian = rotation.t() @ hessian @ rotation
            if cross_hessian is not None:
                cross_hessian = rotation.t() @ cross_hessian @ rotation

        if not quantizer.ready():
            quantizer.find_params(weights, weight=True)

        dead = torch.diag(hessian) == 0
        hessian[dead, dead] = 1
        weights[:, dead] = 0
        if cross_hessian is not None:
            cross_hessian[:, dead] = 0

        if actorder:
            permutation = torch.argsort(torch.diag(hessian), descending=True)
            weights = weights[:, permutation]
            hessian = hessian[permutation][:, permutation]
            if cross_hessian is not None:
                cross_hessian = cross_hessian[permutation][:, permutation]
            inverse_permutation = torch.argsort(permutation)

        inverse_factor, _ = gptq_inverse_factor(hessian, percdamp)
        correction = None
        if cross_hessian is not None:
            # Strictly upper-triangular asymmetric-calibration correction,
            # exactly the official GPTAQ construction:
            # P = alpha * triu(dXXT Hinv^T, 1) Hinv.
            correction = float(gptaq_alpha) * (
                (cross_hessian @ inverse_factor.t()).triu_(diagonal=1)
                @ inverse_factor
            )
        quantized_weights = torch.zeros_like(weights)

        for block_start in range(0, hessian.shape[0], blocksize):
            block_end = min(block_start + blocksize, hessian.shape[0])
            count = block_end - block_start
            block_weights = weights[:, block_start:block_end].clone()
            block_quantized = torch.zeros_like(block_weights)
            block_errors = torch.zeros_like(block_weights)
            block_inverse = inverse_factor[
                block_start:block_end,
                block_start:block_end,
            ]
            block_correction = None
            if correction is not None:
                block_correction = correction[
                    block_start:block_end,
                    block_start:block_end,
                ]

            for column in range(count):
                weight_column = block_weights[:, column]
                diagonal = block_inverse[column, column]
                global_column = block_start + column

                if groupsize != -1 and global_column % groupsize == 0:
                    quantizer.find_params(
                        weights[:, global_column : global_column + groupsize],
                        weight=True,
                    )

                quantized_column = quantize(
                    weight_column.unsqueeze(1),
                    quantizer.scale,
                    quantizer.zero,
                    quantizer.maxq,
                ).flatten()
                block_quantized[:, column] = quantized_column

                error = (weight_column - quantized_column) / diagonal
                update = error.unsqueeze(1) @ block_inverse[
                    column,
                    column:,
                ].unsqueeze(0)
                if block_correction is not None:
                    update = update - weight_column.unsqueeze(
                        1
                    ) @ block_correction[column, column:].unsqueeze(0)
                block_weights[:, column:] -= update
                block_errors[:, column] = error

            quantized_weights[:, block_start:block_end] = block_quantized
            tail_update = (
                block_errors @ inverse_factor[block_start:block_end, block_end:]
            )
            if correction is not None:
                # After the scan block_weights holds the quantized columns,
                # matching the official W1 in the block-level update.
                tail_update = tail_update - block_weights @ correction[
                    block_start:block_end,
                    block_end:,
                ]
            weights[:, block_end:] -= tail_update

        if actorder:
            quantized_weights = quantized_weights[:, inverse_permutation]
        if rotation is not None:
            quantized_weights = quantized_weights @ rotation.t()
        if isinstance(layer, transformers.Conv1D):
            quantized_weights = quantized_weights.t()

        result = quantized_weights.reshape(layer.weight.shape).to(layer.weight.dtype)
        if return_W:
            return result
        layer.weight.data.copy_(result)
        return None

    def run_weight_correct(
        self,
        layer,
        percdamp: float = 0.01,
        perccorr: float = 0.25,
    ) -> None:
        weights = layer.weight.detach().clone()
        if isinstance(layer, nn.Conv2d):
            weights = weights.flatten(1)
        if isinstance(layer, transformers.Conv1D):
            weights = weights.t()
        weights = weights.float()

        hessian = self.H_q.clone()
        dead = torch.diag(hessian) == 0
        hessian[dead, dead] = 1
        weights[:, dead] = 0
        hessian_inverse, _ = damped_hessian_inverse(hessian, percdamp)
        weights += (weights @ self.H_delta @ hessian_inverse) * float(perccorr)

        if isinstance(layer, transformers.Conv1D):
            weights = weights.t()
        layer.weight.data.copy_(
            weights.reshape(layer.weight.shape).to(layer.weight.dtype)
        )

    def qep_output_reconstruction_surrogate(
        self,
        layer,
        candidate_weight: torch.Tensor,
        reference_weight: torch.Tensor,
    ) -> float:
        candidate = candidate_weight.detach().float()
        reference = reference_weight.detach().float()
        if isinstance(layer, nn.Conv2d):
            candidate = candidate.flatten(1)
            reference = reference.flatten(1)
        if isinstance(layer, transformers.Conv1D):
            candidate = candidate.t()
            reference = reference.t()
        delta_weight = candidate - reference
        quadratic = ((delta_weight @ self.H_q) * delta_weight).sum()
        cross = ((delta_weight @ self.H_delta.t()) * reference).sum()
        return float((quadratic - 2.0 * cross).item())

    def run_gptq_qep_candidates(
        self,
        layer,
        candidates,
        percdampqep: float,
        percdamp: float,
        wbits: int,
        groupsize: int,
        actorder: bool,
    ):
        reference_weight = layer.weight.detach().clone()
        best_weight = None
        best_alpha = None
        best_score = None
        scores = {}
        try:
            for alpha in candidates:
                alpha = float(alpha)
                layer.weight.data.copy_(reference_weight)
                if alpha != 0.0:
                    self.run_weight_correct(
                        layer,
                        percdamp=percdampqep,
                        perccorr=alpha,
                    )
                candidate_weight = self.run_gptq(
                    layer,
                    percdamp=percdamp,
                    wbits=wbits,
                    groupsize=groupsize,
                    actorder=actorder,
                    return_W=True,
                )
                score = self.qep_output_reconstruction_surrogate(
                    layer,
                    candidate_weight,
                    reference_weight,
                )
                scores[alpha] = score
                if best_score is None or score < best_score:
                    best_score = score
                    best_alpha = alpha
                    best_weight = candidate_weight.detach().clone()
        finally:
            layer.weight.data.copy_(reference_weight)
        return best_weight, best_alpha, scores

    def free(self) -> None:
        for attribute in ("H_q", "H_delta"):
            if hasattr(self, attribute):
                delattr(self, attribute)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
