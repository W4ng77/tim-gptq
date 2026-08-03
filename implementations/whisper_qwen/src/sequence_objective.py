"""Sequence-level counterfactual objectives for mixed-precision routing."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


def paired_nll_recovery(plain: Mapping, restored: Mapping) -> dict:
    """Measure token-NLL recovered by a counterfactual FP16 restoration.

    Positive values mean that restoring the candidate module reduced
    teacher-forced sequence loss relative to the uniformly quantized model.
    """
    plain_rows = plain["per_example"]
    restored_rows = restored["per_example"]
    if len(plain_rows) != len(restored_rows):
        raise ValueError("Paired NLL scores have different example counts.")

    paired = []
    total_delta = 0.0
    total_tokens = 0
    for left, right in zip(plain_rows, restored_rows):
        if left["example_id"] != right["example_id"]:
            raise ValueError(
                "Paired NLL example IDs differ: "
                f"{left['example_id']!r} != {right['example_id']!r}"
            )
        if int(left["num_tokens"]) != int(right["num_tokens"]):
            raise ValueError(
                f"Paired token counts differ for {left['example_id']!r}."
            )
        num_tokens = int(left["num_tokens"])
        mean_delta = float(left["mean_nll"]) - float(right["mean_nll"])
        total_nll_delta = mean_delta * num_tokens
        total_delta += total_nll_delta
        total_tokens += num_tokens
        paired.append(
            {
                "example_id": left["example_id"],
                "num_tokens": num_tokens,
                "plain_mean_nll": float(left["mean_nll"]),
                "restored_mean_nll": float(right["mean_nll"]),
                "mean_nll_recovery": mean_delta,
                "total_nll_recovery": total_nll_delta,
            }
        )

    return {
        "num_examples": len(paired),
        "num_tokens": total_tokens,
        "mean_token_nll_recovery": total_delta / max(total_tokens, 1),
        "num_examples_improved": sum(
            row["mean_nll_recovery"] > 0.0 for row in paired
        ),
        "per_example": paired,
    }


def paired_example_bootstrap_ci(
    paired_rows: Sequence[Mapping],
    *,
    confidence: float = 0.95,
    replicates: int = 10_000,
    seed: int = 0,
) -> dict:
    """Bootstrap the paired mean-token-NLL recovery over examples."""
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1).")
    if int(replicates) <= 0:
        raise ValueError("replicates must be positive.")
    if not paired_rows:
        raise ValueError("At least one paired row is required.")

    deltas = np.asarray(
        [float(row["total_nll_recovery"]) for row in paired_rows],
        dtype=np.float64,
    )
    tokens = np.asarray(
        [int(row["num_tokens"]) for row in paired_rows],
        dtype=np.float64,
    )
    rng = np.random.default_rng(int(seed))
    samples = np.empty(int(replicates), dtype=np.float64)
    n = len(paired_rows)
    for start in range(0, int(replicates), 1_000):
        size = min(1_000, int(replicates) - start)
        indices = rng.integers(0, n, size=(size, n))
        sampled_deltas = deltas[indices].sum(axis=1)
        sampled_tokens = tokens[indices].sum(axis=1)
        samples[start : start + size] = sampled_deltas / np.maximum(
            sampled_tokens, 1.0
        )

    tail = (1.0 - float(confidence)) / 2.0
    point = float(deltas.sum() / max(tokens.sum(), 1.0))
    return {
        "point_estimate": point,
        "confidence": float(confidence),
        "lower": float(np.quantile(samples, tail)),
        "upper": float(np.quantile(samples, 1.0 - tail)),
        "replicates": int(replicates),
        "seed": int(seed),
        "resampling_unit": "example",
    }


def _word_edit_distance(reference: str, hypothesis: str) -> int:
    reference_words = str(reference).split()
    hypothesis_words = str(hypothesis).split()
    previous = list(range(len(hypothesis_words) + 1))
    for row, reference_word in enumerate(reference_words, start=1):
        current = [row]
        for column, hypothesis_word in enumerate(hypothesis_words, start=1):
            current.append(
                min(
                    previous[column] + 1,
                    current[column - 1] + 1,
                    previous[column - 1]
                    + int(reference_word != hypothesis_word),
                )
            )
        previous = current
    return previous[-1]


def paired_transcript_drift_recovery(
    fp16_score: Mapping,
    plain_score: Mapping,
    restored_score: Mapping,
) -> dict:
    """Measure autoregressive transcript drift removed by FP16 restoration."""
    fp_ids = fp16_score["example_ids"]
    plain_ids = plain_score["example_ids"]
    restored_ids = restored_score["example_ids"]
    if fp_ids != plain_ids or fp_ids != restored_ids:
        raise ValueError("Transcript-drift example IDs are not paired.")

    paired = []
    total_recovery = 0
    total_reference_words = 0
    for example_id, fp_text, plain_text, restored_text in zip(
        fp_ids,
        fp16_score["predictions"],
        plain_score["predictions"],
        restored_score["predictions"],
    ):
        reference_words = max(len(str(fp_text).split()), 1)
        plain_errors = _word_edit_distance(fp_text, plain_text)
        restored_errors = _word_edit_distance(fp_text, restored_text)
        recovery = plain_errors - restored_errors
        total_recovery += recovery
        total_reference_words += reference_words
        paired.append(
            {
                "example_id": example_id,
                "reference_words": reference_words,
                "plain_drift_errors": plain_errors,
                "restored_drift_errors": restored_errors,
                "total_drift_error_recovery": recovery,
            }
        )

    return {
        "num_examples": len(paired),
        "num_reference_words": total_reference_words,
        "sequence_drift_recovery": (
            total_recovery / max(total_reference_words, 1)
        ),
        "plain_fp16_drift": (
            sum(row["plain_drift_errors"] for row in paired)
            / max(total_reference_words, 1)
        ),
        "restored_fp16_drift": (
            sum(row["restored_drift_errors"] for row in paired)
            / max(total_reference_words, 1)
        ),
        "num_examples_improved": sum(
            row["total_drift_error_recovery"] > 0 for row in paired
        ),
        "per_example": paired,
    }


def paired_transcript_bootstrap_ci(
    paired_rows: Sequence[Mapping],
    *,
    confidence: float = 0.95,
    replicates: int = 10_000,
    seed: int = 0,
) -> dict:
    """Bootstrap transcript-drift recovery over paired examples."""
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1).")
    if int(replicates) <= 0:
        raise ValueError("replicates must be positive.")
    if not paired_rows:
        raise ValueError("At least one paired row is required.")

    recoveries = np.asarray(
        [int(row["total_drift_error_recovery"]) for row in paired_rows],
        dtype=np.float64,
    )
    words = np.asarray(
        [int(row["reference_words"]) for row in paired_rows],
        dtype=np.float64,
    )
    rng = np.random.default_rng(int(seed))
    samples = np.empty(int(replicates), dtype=np.float64)
    n = len(paired_rows)
    for start in range(0, int(replicates), 1_000):
        size = min(1_000, int(replicates) - start)
        indices = rng.integers(0, n, size=(size, n))
        samples[start : start + size] = (
            recoveries[indices].sum(axis=1)
            / np.maximum(words[indices].sum(axis=1), 1.0)
        )

    tail = (1.0 - float(confidence)) / 2.0
    return {
        "point_estimate": float(recoveries.sum() / max(words.sum(), 1.0)),
        "confidence": float(confidence),
        "lower": float(np.quantile(samples, tail)),
        "upper": float(np.quantile(samples, 1.0 - tail)),
        "replicates": int(replicates),
        "seed": int(seed),
        "resampling_unit": "example",
    }


def select_best_sequence_candidate(
    candidate_scores: Mapping[str, Mapping],
    *,
    metric: str = "mean_token_nll_recovery",
) -> dict:
    """Select the largest positive sequence-NLL recovery for confirmation."""
    if not candidate_scores:
        raise ValueError("At least one sequence candidate is required.")
    ranked = sorted(
        candidate_scores,
        key=lambda name: (
            -float(candidate_scores[name][metric]),
            name,
        ),
    )
    candidate = ranked[0]
    recovery = float(candidate_scores[candidate][metric])
    return {
        "candidate": candidate,
        "selection_recovery": recovery,
        "selection_metric": metric,
        "eligible_for_confirmation": recovery > 0.0,
        "ranking": ranked,
    }
