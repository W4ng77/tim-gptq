#!/usr/bin/env python3
"""Paired stratified WER bootstrap for one candidate against one baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from rapidfuzz.distance import Levenshtein


DEFAULT_DATASETS = (
    "librispeech-clean",
    "librispeech-other",
    "spgispeech",
    "voxpopuli",
    "gigaspeech",
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-run",
        type=Path,
        action="append",
        required=True,
        help="Repeat when baseline datasets are split across run directories.",
    )
    parser.add_argument(
        "--candidate-run",
        type=Path,
        action="append",
        required=True,
        help="Repeat when candidate datasets are split across run directories.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    parser.add_argument("--reps", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument("--batch-size", type=int, default=256)
    return parser.parse_args()


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def unique_dataset_file(run_dirs, dataset, side):
    matches = [path / f"{dataset}.jsonl" for path in run_dirs]
    matches = [path for path in matches if path.is_file()]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one {side} file for {dataset}, got {matches}"
        )
    return matches[0]


def aligned_errors(baseline_path: Path, candidate_path: Path):
    baseline = read_jsonl(baseline_path)
    candidate = read_jsonl(candidate_path)
    if len(baseline) != len(candidate):
        raise ValueError(f"Row count mismatch: {baseline_path} / {candidate_path}")
    lengths = []
    baseline_errors = []
    candidate_errors = []
    for left, right in zip(baseline, candidate, strict=True):
        if left["example_id"] != right["example_id"]:
            raise ValueError(
                f"Example alignment mismatch: {left['example_id']} / "
                f"{right['example_id']}"
            )
        if left["reference"] != right["reference"]:
            raise ValueError(f"Reference mismatch for {left['example_id']}")
        reference = left["reference"].split()
        lengths.append(len(reference))
        baseline_errors.append(
            Levenshtein.distance(reference, left["prediction"].split())
        )
        candidate_errors.append(
            Levenshtein.distance(reference, right["prediction"].split())
        )
    return (
        np.asarray(lengths, dtype=np.int64),
        np.asarray(baseline_errors, dtype=np.int64),
        np.asarray(candidate_errors, dtype=np.int64),
    )


def main():
    args = parse_args()
    if args.reps < 100 or args.batch_size < 1:
        raise ValueError("reps must be >=100 and batch-size must be positive")
    data = {}
    observed = {}
    for dataset in args.datasets:
        baseline_path = unique_dataset_file(
            args.baseline_run, dataset, "baseline"
        )
        candidate_path = unique_dataset_file(
            args.candidate_run, dataset, "candidate"
        )
        lengths, baseline_errors, candidate_errors = aligned_errors(
            baseline_path, candidate_path
        )
        denominator = int(lengths.sum())
        baseline_wer = float(baseline_errors.sum() / denominator)
        candidate_wer = float(candidate_errors.sum() / denominator)
        data[dataset] = (lengths, baseline_errors, candidate_errors)
        observed[dataset] = {
            "examples": int(len(lengths)),
            "reference_words": denominator,
            "baseline_wer": baseline_wer,
            "candidate_wer": candidate_wer,
            "delta": candidate_wer - baseline_wer,
        }

    rng = np.random.default_rng(args.seed)
    bootstrap_by_dataset = {
        dataset: np.empty(args.reps, dtype=np.float64)
        for dataset in args.datasets
    }
    for dataset, (lengths, baseline_errors, candidate_errors) in data.items():
        n = len(lengths)
        for start in range(0, args.reps, args.batch_size):
            stop = min(start + args.batch_size, args.reps)
            draw = rng.integers(0, n, size=(stop - start, n), dtype=np.int32)
            denominator = lengths[draw].sum(axis=1)
            baseline_wer = baseline_errors[draw].sum(axis=1) / denominator
            candidate_wer = candidate_errors[draw].sum(axis=1) / denominator
            bootstrap_by_dataset[dataset][start:stop] = (
                candidate_wer - baseline_wer
            )

    macro_samples = np.mean(
        np.stack([bootstrap_by_dataset[name] for name in args.datasets]),
        axis=0,
    )
    macro_estimate = float(
        np.mean([observed[name]["delta"] for name in args.datasets])
    )

    def summarize(samples, estimate):
        centered = samples - estimate
        exceedances = int(np.count_nonzero(np.abs(centered) >= abs(estimate)))
        return {
            "estimate": float(estimate),
            "ci95_lower": float(np.quantile(samples, 0.025)),
            "ci95_upper": float(np.quantile(samples, 0.975)),
            "p_two_sided_centered": (
                (exceedances + 1.0) / (len(samples) + 1.0)
            ),
        }

    result = {
        "baseline_runs": [str(path.resolve()) for path in args.baseline_run],
        "candidate_runs": [str(path.resolve()) for path in args.candidate_run],
        "datasets": list(args.datasets),
        "reps": args.reps,
        "seed": args.seed,
        "observed": observed,
        "bootstrap": {
            dataset: summarize(
                bootstrap_by_dataset[dataset], observed[dataset]["delta"]
            )
            for dataset in args.datasets
        },
        "macro": {
            "baseline_wer": float(
                np.mean([observed[name]["baseline_wer"] for name in args.datasets])
            ),
            "candidate_wer": float(
                np.mean([observed[name]["candidate_wer"] for name in args.datasets])
            ),
            **summarize(macro_samples, macro_estimate),
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "paired_bootstrap.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")

    macro = result["macro"]
    lines = [
        "# Paired stratified WER bootstrap",
        "",
        f"Replicates: {args.reps:,}; seed: `{args.seed}`. Resampling is paired "
        "within each dataset; macro is the unweighted mean of dataset corpus WER.",
        "",
        "| Scope | Baseline WER % | Candidate WER % | Δ pp [95% CI] | p |",
        "|---|---:|---:|---:|---:|",
    ]
    for dataset in args.datasets:
        row = observed[dataset]
        boot = result["bootstrap"][dataset]
        lines.append(
            f"| {dataset} | {100*row['baseline_wer']:.3f} | "
            f"{100*row['candidate_wer']:.3f} | {100*boot['estimate']:+.3f} "
            f"[{100*boot['ci95_lower']:+.3f}, "
            f"{100*boot['ci95_upper']:+.3f}] | "
            f"{boot['p_two_sided_centered']:.4f} |"
        )
    lines.append(
        f"| **macro** | **{100*macro['baseline_wer']:.3f}** | "
        f"**{100*macro['candidate_wer']:.3f}** | "
        f"**{100*macro['estimate']:+.3f} "
        f"[{100*macro['ci95_lower']:+.3f}, "
        f"{100*macro['ci95_upper']:+.3f}]** | "
        f"**{macro['p_two_sided_centered']:.4f}** |"
    )
    lines.extend(
        [
            "",
            "Negative Δ favors the candidate. The p-value uses the centered "
            "paired bootstrap null with a plus-one correction.",
            "",
        ]
    )
    (args.output_dir / "PAIRED_BOOTSTRAP_REPORT.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
