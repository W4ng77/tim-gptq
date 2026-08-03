#!/usr/bin/env python3
"""Compare held-out greedy predictions against matched FP16 trajectories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from rapidfuzz.distance import Levenshtein


COLLAPSE_RATE_LIMIT = 0.05
ONE_SIDED_95_Z = 1.6448536269514722


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pair",
        nargs=3,
        action="append",
        required=True,
        metavar=("LABEL", "FP16_RUN", "CANDIDATE_RUN"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_score(run_dir: Path):
    path = run_dir / "calibration_wer.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    keys = ("example_ids", "references", "predictions")
    lengths = {len(payload[key]) for key in keys}
    if len(lengths) != 1:
        raise ValueError(f"Misaligned calibration score arrays: {path}")
    rows = {}
    for example_id, reference, prediction in zip(
        payload["example_ids"],
        payload["references"],
        payload["predictions"],
        strict=True,
    ):
        if example_id in rows:
            raise ValueError(f"Duplicate example id {example_id!r}: {path}")
        rows[example_id] = (reference, prediction)
    return rows


def common_prefix_length(left, right):
    count = 0
    for a, b in zip(left, right):
        if a != b:
            break
        count += 1
    return count


def wilson_upper_bound(successes, trials, z=ONE_SIDED_95_Z):
    if trials <= 0:
        return 1.0
    rate = successes / trials
    z2 = z * z
    numerator = (
        rate
        + z2 / (2 * trials)
        + z
        * np.sqrt(
            rate * (1 - rate) / trials
            + z2 / (4 * trials * trials)
        )
    )
    return float(numerator / (1 + z2 / trials))


def analyze(anchor_rows, candidate_rows):
    if set(anchor_rows) != set(candidate_rows):
        raise ValueError("FP16 and candidate example ids differ")

    anchor_errors = 0
    candidate_errors = 0
    reference_words = 0
    trajectory_edits = 0
    anchor_words = 0
    exact = 0
    prefix_words = 0
    length_ratios = []
    overgenerated = 0
    truncated = 0
    flips = 0
    collapses = 0

    for example_id in anchor_rows:
        reference, anchor_prediction = anchor_rows[example_id]
        candidate_reference, candidate_prediction = candidate_rows[example_id]
        if reference != candidate_reference:
            raise ValueError(f"Reference mismatch for {example_id}")

        ref = reference.split()
        anchor = anchor_prediction.split()
        candidate = candidate_prediction.split()
        ref_len = len(ref)
        anchor_len = len(anchor)

        anchor_error = Levenshtein.distance(ref, anchor)
        candidate_error = Levenshtein.distance(ref, candidate)
        anchor_errors += anchor_error
        candidate_errors += candidate_error
        reference_words += ref_len
        trajectory_edits += Levenshtein.distance(anchor, candidate)
        anchor_words += anchor_len
        prefix_words += common_prefix_length(anchor, candidate)
        exact += int(anchor == candidate)
        is_flip = candidate_error - anchor_error > 0.5 * max(ref_len, 1)
        flips += int(is_flip)

        if anchor_len == 0:
            ratio = 1.0 if not candidate else float("inf")
        else:
            ratio = len(candidate) / anchor_len
        length_ratios.append(ratio)
        is_overgenerated = ratio > 2.0
        is_truncated = ratio < 0.5
        overgenerated += int(is_overgenerated)
        truncated += int(is_truncated)
        collapses += int(is_flip or is_overgenerated or is_truncated)

    finite_ratios = np.asarray(
        [ratio for ratio in length_ratios if np.isfinite(ratio)],
        dtype=np.float64,
    )
    n = len(anchor_rows)
    collapse_upper = wilson_upper_bound(collapses, n)
    return {
        "num_examples": n,
        "anchor_task_wer": anchor_errors / max(reference_words, 1),
        "candidate_task_wer": candidate_errors / max(reference_words, 1),
        "task_wer_delta": (
            candidate_errors - anchor_errors
        ) / max(reference_words, 1),
        "trajectory_edit_rate": trajectory_edits / max(anchor_words, 1),
        "exact_trajectory_fraction": exact / max(n, 1),
        "word_prefix_retention": prefix_words / max(anchor_words, 1),
        "flip_fraction": flips / max(n, 1),
        "overgeneration_fraction": overgenerated / max(n, 1),
        "truncation_fraction": truncated / max(n, 1),
        "collapse_fraction": collapses / max(n, 1),
        "collapse_rate_upper_95": collapse_upper,
        "stability_gate_limit": COLLAPSE_RATE_LIMIT,
        "stability_gate_pass": collapse_upper <= COLLAPSE_RATE_LIMIT,
        "length_ratio_p10": float(np.quantile(finite_ratios, 0.10)),
        "length_ratio_p50": float(np.quantile(finite_ratios, 0.50)),
        "length_ratio_p90": float(np.quantile(finite_ratios, 0.90)),
        "nonfinite_length_ratios": len(length_ratios) - len(finite_ratios),
    }


def main():
    args = parse_args()
    results = {}
    sources = {}
    for label, anchor, candidate in args.pair:
        if label in results:
            raise ValueError(f"Duplicate label: {label}")
        anchor_path = Path(anchor)
        candidate_path = Path(candidate)
        results[label] = analyze(
            load_score(anchor_path),
            load_score(candidate_path),
        )
        sources[label] = {
            "anchor": str(anchor_path.resolve()),
            "candidate": str(candidate_path.resolve()),
        }

    payload = {
        "collapse_definition": (
            "candidate-vs-FP16 extra word errors > 0.5 * reference words "
            "or prediction length ratio outside [0.5, 2]"
        ),
        "gate": (
            "pass iff the one-sided 95% Wilson upper confidence bound on "
            f"held-out collapse rate is <= {COLLAPSE_RATE_LIMIT:.3f}"
        ),
        "sources": sources,
        "results": results,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "trajectory_stability.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# Held-out greedy trajectory stability",
        "",
        "All candidate predictions are aligned to a matched FP16 prediction on "
        "the same held-out calibration audio. Trajectory edit/prefix metrics "
        "use FP16 prediction words as the denominator.",
        "",
        "A collapse is an extra error exceeding half the reference or a "
        "candidate/FP16 length ratio outside `[0.5, 2]`. The frozen gate "
        f"passes when its one-sided 95% Wilson UCB is ≤"
        f"`{100*COLLAPSE_RATE_LIMIT:.1f}%`.",
        "",
        "| Candidate | Task WER % | Δ vs FP16 pp | Trajectory edit % | "
        "Prefix retained % | Collapse % | UCB95 % | Gate |",
        "|---|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for label, row in results.items():
        lines.append(
            f"| {label} | {100*row['candidate_task_wer']:.3f} | "
            f"{100*row['task_wer_delta']:+.3f} | "
            f"{100*row['trajectory_edit_rate']:.3f} | "
            f"{100*row['word_prefix_retention']:.2f} | "
            f"{100*row['collapse_fraction']:.2f} | "
            f"{100*row['collapse_rate_upper_95']:.2f} | "
            f"{'PASS' if row['stability_gate_pass'] else 'FAIL'} |"
        )
    lines.append("")
    (args.output_dir / "TRAJECTORY_STABILITY_REPORT.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
