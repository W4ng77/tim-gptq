#!/usr/bin/env python3
"""Crossed calibration-draw × utterance bootstrap for a 2×2 WER interaction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from paired_multidraw_bootstrap import summarize
from paired_multidraw_factorial import load_quad
from paired_run_bootstrap import DEFAULT_DATASETS


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
    parser.add_argument("--reps", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument("--batch-size", type=int, default=128)
    return parser.parse_args()


def main():
    args = parse_args()
    if len(args.quad) < 2:
        raise ValueError("Crossed inference requires at least two calibration draws")
    if args.reps < 100 or args.batch_size < 1:
        raise ValueError("reps must be >=100 and batch-size must be positive")
    quads = [tuple(Path(value) for value in row) for row in args.quad]
    data = {
        dataset: [load_quad(quad, dataset) for quad in quads]
        for dataset in args.datasets
    }

    observed = {}
    draw_macro_interactions = np.zeros(len(quads), dtype=np.float64)
    for dataset, draws in data.items():
        lengths0 = draws[0][0]
        for lengths, *_ in draws[1:]:
            if not np.array_equal(lengths0, lengths):
                raise ValueError(f"Reference lengths differ across draws: {dataset}")
        interactions = np.asarray(
            [
                (
                    errors11.sum()
                    - errors10.sum()
                    - errors01.sum()
                    + errors00.sum()
                )
                / lengths.sum()
                for lengths, errors00, errors10, errors01, errors11 in draws
            ],
            dtype=np.float64,
        )
        observed[dataset] = {
            "interaction_draw_mean": float(interactions.mean()),
            "interaction_by_draw": [float(value) for value in interactions],
        }
        draw_macro_interactions += interactions / len(args.datasets)

    rng = np.random.default_rng(args.seed)
    boot = {
        dataset: np.empty(args.reps, dtype=np.float64)
        for dataset in args.datasets
    }
    draw_count = len(quads)
    for start in range(0, args.reps, args.batch_size):
        stop = min(start + args.batch_size, args.reps)
        size = stop - start
        draw_indices = rng.integers(
            0, draw_count, size=(size, draw_count), dtype=np.int32
        )
        for dataset, draws in data.items():
            lengths = draws[0][0]
            utterance_indices = rng.integers(
                0, len(lengths), size=(size, len(lengths)), dtype=np.int32
            )
            denominator = lengths[utterance_indices].sum(axis=1)
            draw_samples = np.empty((size, draw_count), dtype=np.float64)
            for draw_index, (_, e00, e10, e01, e11) in enumerate(draws):
                draw_samples[:, draw_index] = (
                    e11[utterance_indices].sum(axis=1)
                    - e10[utterance_indices].sum(axis=1)
                    - e01[utterance_indices].sum(axis=1)
                    + e00[utterance_indices].sum(axis=1)
                ) / denominator
            boot[dataset][start:stop] = np.take_along_axis(
                draw_samples, draw_indices, axis=1
            ).mean(axis=1)

    macro_samples = np.mean(
        np.stack([boot[dataset] for dataset in args.datasets]), axis=0
    )
    macro_estimate = float(
        np.mean([row["interaction_draw_mean"] for row in observed.values()])
    )
    result = {
        "inference": "crossed_calibration_draw_by_utterance_bootstrap",
        "draws": draw_count,
        "factor_a": args.factor_a,
        "factor_b": args.factor_b,
        "datasets": list(args.datasets),
        "reps": args.reps,
        "seed": args.seed,
        "observed": observed,
        "macro_interaction_by_draw": [
            float(value) for value in draw_macro_interactions
        ],
        "bootstrap": {
            dataset: summarize(
                boot[dataset], observed[dataset]["interaction_draw_mean"]
            )
            for dataset in args.datasets
        },
        "macro": summarize(macro_samples, macro_estimate),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "crossed_multidraw_factorial.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    lines = [
        "# Crossed calibration-draw × utterance 2×2 interaction bootstrap",
        "",
        f"{draw_count} calibration draws; factors: **{args.factor_a} × "
        f"{args.factor_b}**; {args.reps:,} replicates; seed `{args.seed}`.",
        "Each replicate resamples calibration draws and evaluation utterances; "
        "draw indices are shared across domains.",
        "",
        "| Scope | Interaction pp [95% CI] | p |",
        "|---|---:|---:|",
    ]
    for dataset in args.datasets:
        stat = result["bootstrap"][dataset]
        lines.append(
            f"| {dataset} | {100*stat['estimate']:+.3f} "
            f"[{100*stat['ci95_lower']:+.3f}, {100*stat['ci95_upper']:+.3f}] | "
            f"{stat['p_two_sided_centered']:.4f} |"
        )
    macro = result["macro"]
    lines.append(
        f"| **macro** | **{100*macro['estimate']:+.3f} "
        f"[{100*macro['ci95_lower']:+.3f}, {100*macro['ci95_upper']:+.3f}]** | "
        f"**{macro['p_two_sided_centered']:.4f}** |"
    )
    lines.extend(
        [
            "",
            "Macro interaction by calibration draw (pp): `"
            + "/".join(f"{100*value:+.3f}" for value in draw_macro_interactions)
            + "`.",
            "",
            "With few calibration draws this interval is intentionally wider "
            "than the fixed-draw utterance bootstrap and remains descriptive.",
            "",
        ]
    )
    (args.output_dir / "CROSSED_MULTIDRAW_FACTORIAL_REPORT.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
