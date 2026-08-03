#!/usr/bin/env python3
"""Paired stratified bootstrap for a 2x2 WER factorial interaction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from paired_run_bootstrap import DEFAULT_DATASETS, aligned_errors


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corner-00", type=Path, action="append", required=True)
    parser.add_argument("--corner-10", type=Path, action="append", required=True)
    parser.add_argument("--corner-01", type=Path, action="append", required=True)
    parser.add_argument("--corner-11", type=Path, action="append", required=True)
    parser.add_argument("--factor-a", default="A")
    parser.add_argument("--factor-b", default="B")
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reps", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument("--batch-size", type=int, default=256)
    return parser.parse_args()


def summarize(samples, estimate):
    centered = samples - estimate
    count = int(np.count_nonzero(np.abs(centered) >= abs(estimate)))
    return {
        "estimate": float(estimate),
        "ci95_lower": float(np.quantile(samples, 0.025)),
        "ci95_upper": float(np.quantile(samples, 0.975)),
        "p_two_sided_centered": (count + 1.0) / (len(samples) + 1.0),
    }


def locate(run_dirs, dataset: str, corner: str):
    matches = [
        path
        for path in (run_dir / f"{dataset}.jsonl" for run_dir in run_dirs)
        if path.is_file()
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one corner-{corner} file for {dataset}: {matches}"
        )
    return matches[0]


def main():
    args = parse_args()
    if args.reps < 100 or args.batch_size < 1:
        raise ValueError("reps must be >=100 and batch-size must be positive")
    data = {}
    observed = {}
    for dataset in args.datasets:
        path00 = locate(args.corner_00, dataset, "00")
        path10 = locate(args.corner_10, dataset, "10")
        path01 = locate(args.corner_01, dataset, "01")
        path11 = locate(args.corner_11, dataset, "11")
        lengths, errors00, errors10 = aligned_errors(
            path00, path10
        )
        lengths01, errors00_01, errors01 = aligned_errors(
            path00, path01
        )
        lengths11, errors00_11, errors11 = aligned_errors(
            path00, path11
        )
        if not (
            np.array_equal(lengths, lengths01)
            and np.array_equal(lengths, lengths11)
            and np.array_equal(errors00, errors00_01)
            and np.array_equal(errors00, errors00_11)
        ):
            raise ValueError(f"Corner alignment differs for {dataset}")
        denominator = int(lengths.sum())
        interaction = float(
            (errors11.sum() - errors10.sum() - errors01.sum() + errors00.sum())
            / denominator
        )
        data[dataset] = (
            lengths,
            errors00,
            errors10,
            errors01,
            errors11,
        )
        observed[dataset] = {
            "examples": int(len(lengths)),
            "reference_words": denominator,
            "wer_00": float(errors00.sum() / denominator),
            "wer_10": float(errors10.sum() / denominator),
            "wer_01": float(errors01.sum() / denominator),
            "wer_11": float(errors11.sum() / denominator),
            "interaction": interaction,
        }

    rng = np.random.default_rng(args.seed)
    boot = {
        dataset: np.empty(args.reps, dtype=np.float64)
        for dataset in args.datasets
    }
    for dataset, arrays in data.items():
        lengths, errors00, errors10, errors01, errors11 = arrays
        n = len(lengths)
        for start in range(0, args.reps, args.batch_size):
            stop = min(start + args.batch_size, args.reps)
            indices = rng.integers(
                0, n, size=(stop - start, n), dtype=np.int32
            )
            denominator = lengths[indices].sum(axis=1)
            boot[dataset][start:stop] = (
                errors11[indices].sum(axis=1)
                - errors10[indices].sum(axis=1)
                - errors01[indices].sum(axis=1)
                + errors00[indices].sum(axis=1)
            ) / denominator

    macro_samples = np.mean(
        np.stack([boot[name] for name in args.datasets]), axis=0
    )
    macro_estimate = float(
        np.mean([observed[name]["interaction"] for name in args.datasets])
    )
    result = {
        "factor_a": args.factor_a,
        "factor_b": args.factor_b,
        "datasets": list(args.datasets),
        "reps": args.reps,
        "seed": args.seed,
        "corners": {
            "00": [str(path.resolve()) for path in args.corner_00],
            "10": [str(path.resolve()) for path in args.corner_10],
            "01": [str(path.resolve()) for path in args.corner_01],
            "11": [str(path.resolve()) for path in args.corner_11],
        },
        "observed": observed,
        "bootstrap": {
            name: summarize(boot[name], observed[name]["interaction"])
            for name in args.datasets
        },
        "macro": summarize(macro_samples, macro_estimate),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "paired_factorial_interaction.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# Paired 2×2 WER factorial interaction",
        "",
        f"Factors: **{args.factor_a} × {args.factor_b}**. Interaction is "
        "`WER11 − WER10 − WER01 + WER00`; negative values mean the joint "
        "degradation is smaller than the additive prediction.",
        "",
        "| Scope | Interaction pp [95% CI] | p |",
        "|---|---:|---:|",
    ]
    for name in args.datasets:
        stat = result["bootstrap"][name]
        lines.append(
            f"| {name} | {100*stat['estimate']:+.3f} "
            f"[{100*stat['ci95_lower']:+.3f}, "
            f"{100*stat['ci95_upper']:+.3f}] | "
            f"{stat['p_two_sided_centered']:.4f} |"
        )
    macro = result["macro"]
    lines.append(
        f"| **macro** | **{100*macro['estimate']:+.3f} "
        f"[{100*macro['ci95_lower']:+.3f}, "
        f"{100*macro['ci95_upper']:+.3f}]** | "
        f"**{macro['p_two_sided_centered']:.4f}** |"
    )
    lines.append("")
    (args.output_dir / "PAIRED_FACTORIAL_REPORT.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
