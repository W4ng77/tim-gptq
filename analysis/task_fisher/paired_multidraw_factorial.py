#!/usr/bin/env python3
"""Draw-averaged paired bootstrap for a 2x2 WER interaction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from paired_run_bootstrap import DEFAULT_DATASETS, aligned_errors


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quad",
        nargs=4,
        action="append",
        metavar=("CORNER00", "CORNER10", "CORNER01", "CORNER11"),
        required=True,
    )
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


def load_quad(paths, dataset):
    corner00, corner10, corner01, corner11 = paths
    path00 = corner00 / f"{dataset}.jsonl"
    lengths, errors00, errors10 = aligned_errors(
        path00, corner10 / f"{dataset}.jsonl"
    )
    lengths01, errors00_01, errors01 = aligned_errors(
        path00, corner01 / f"{dataset}.jsonl"
    )
    lengths11, errors00_11, errors11 = aligned_errors(
        path00, corner11 / f"{dataset}.jsonl"
    )
    if not (
        np.array_equal(lengths, lengths01)
        and np.array_equal(lengths, lengths11)
        and np.array_equal(errors00, errors00_01)
        and np.array_equal(errors00, errors00_11)
    ):
        raise ValueError(f"Corner alignment differs for {dataset}")
    return lengths, errors00, errors10, errors01, errors11


def main():
    args = parse_args()
    if args.reps < 100 or args.batch_size < 1:
        raise ValueError("reps must be >=100 and batch-size must be positive")
    quads = [tuple(Path(value) for value in row) for row in args.quad]
    data = {
        dataset: [load_quad(quad, dataset) for quad in quads]
        for dataset in args.datasets
    }
    observed = {}
    boot = {
        dataset: np.empty(args.reps, dtype=np.float64)
        for dataset in args.datasets
    }
    rng = np.random.default_rng(args.seed)
    for dataset, draws in data.items():
        lengths0 = draws[0][0]
        for lengths, *_ in draws[1:]:
            if not np.array_equal(lengths0, lengths):
                raise ValueError(f"Reference lengths differ across draws: {dataset}")
        interactions = []
        for lengths, errors00, errors10, errors01, errors11 in draws:
            interactions.append(
                float(
                    (
                        errors11.sum()
                        - errors10.sum()
                        - errors01.sum()
                        + errors00.sum()
                    )
                    / lengths.sum()
                )
            )
        observed[dataset] = float(np.mean(interactions))
        n = len(lengths0)
        for start in range(0, args.reps, args.batch_size):
            stop = min(start + args.batch_size, args.reps)
            indices = rng.integers(
                0, n, size=(stop - start, n), dtype=np.int32
            )
            draw_samples = []
            for lengths, errors00, errors10, errors01, errors11 in draws:
                denominator = lengths[indices].sum(axis=1)
                draw_samples.append(
                    (
                        errors11[indices].sum(axis=1)
                        - errors10[indices].sum(axis=1)
                        - errors01[indices].sum(axis=1)
                        + errors00[indices].sum(axis=1)
                    )
                    / denominator
                )
            boot[dataset][start:stop] = np.mean(
                np.stack(draw_samples), axis=0
            )

    macro_samples = np.mean(
        np.stack([boot[name] for name in args.datasets]), axis=0
    )
    macro_estimate = float(np.mean(list(observed.values())))
    result = {
        "draws": len(quads),
        "factor_a": args.factor_a,
        "factor_b": args.factor_b,
        "datasets": list(args.datasets),
        "reps": args.reps,
        "seed": args.seed,
        "observed": observed,
        "bootstrap": {
            name: summarize(boot[name], observed[name])
            for name in args.datasets
        },
        "macro": summarize(macro_samples, macro_estimate),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "paired_multidraw_factorial.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Draw-averaged paired 2×2 WER interaction",
        "",
        f"{len(quads)} calibration draws; factors: **{args.factor_a} × "
        f"{args.factor_b}**; interaction is "
        "`WER11 − WER10 − WER01 + WER00`.",
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
    (args.output_dir / "PAIRED_MULTIDRAW_FACTORIAL_REPORT.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
