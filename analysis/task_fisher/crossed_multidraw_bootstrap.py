#!/usr/bin/env python3
"""Crossed calibration-draw × utterance bootstrap for a mean WER effect."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from paired_multidraw_bootstrap import locate, summarize
from paired_run_bootstrap import DEFAULT_DATASETS, aligned_errors


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pair",
        nargs=3,
        action="append",
        metavar=("BASELINE", "CANDIDATE_HARD", "CANDIDATE_OTHER"),
    )
    parser.add_argument(
        "--simple-pair",
        nargs=2,
        action="append",
        metavar=("BASELINE", "CANDIDATE"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    parser.add_argument("--reps", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--baseline-label", default="Baseline")
    parser.add_argument("--candidate-label", default="Candidate")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.reps < 100 or args.batch_size < 1:
        raise ValueError("reps must be >=100 and batch-size must be positive")
    pairs = []
    for baseline, hard, other in args.pair or []:
        pairs.append(([Path(baseline)], [Path(hard), Path(other)]))
    for baseline, candidate in args.simple_pair or []:
        pairs.append(([Path(baseline)], [Path(candidate)]))
    if len(pairs) < 2:
        raise ValueError("Crossed inference requires at least two calibration draws")

    data = {dataset: [] for dataset in args.datasets}
    for baseline_dirs, candidate_dirs in pairs:
        for dataset in args.datasets:
            data[dataset].append(
                aligned_errors(
                    locate(baseline_dirs, dataset, "baseline"),
                    locate(candidate_dirs, dataset, "candidate"),
                )
            )

    observed = {}
    draw_macro_deltas = np.zeros(len(pairs), dtype=np.float64)
    for dataset, draws in data.items():
        lengths0 = draws[0][0]
        for lengths, _, _ in draws[1:]:
            if not np.array_equal(lengths0, lengths):
                raise ValueError(f"Reference lengths differ across draws: {dataset}")
        baseline_wers = np.asarray(
            [baseline.sum() / lengths.sum() for lengths, baseline, _ in draws],
            dtype=np.float64,
        )
        candidate_wers = np.asarray(
            [candidate.sum() / lengths.sum() for lengths, _, candidate in draws],
            dtype=np.float64,
        )
        deltas = candidate_wers - baseline_wers
        observed[dataset] = {
            "baseline_wer_draw_mean": float(baseline_wers.mean()),
            "candidate_wer_draw_mean": float(candidate_wers.mean()),
            "delta_draw_mean": float(deltas.mean()),
            "delta_by_draw": [float(value) for value in deltas],
        }
        draw_macro_deltas += deltas / len(args.datasets)

    rng = np.random.default_rng(args.seed)
    boot = {
        dataset: np.empty(args.reps, dtype=np.float64)
        for dataset in args.datasets
    }
    draw_count = len(pairs)
    for start in range(0, args.reps, args.batch_size):
        stop = min(start + args.batch_size, args.reps)
        size = stop - start
        # The calibration draw is a shared experimental factor across domains.
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
            for draw_index, (_, baseline_errors, candidate_errors) in enumerate(draws):
                draw_samples[:, draw_index] = (
                    candidate_errors[utterance_indices].sum(axis=1)
                    - baseline_errors[utterance_indices].sum(axis=1)
                ) / denominator
            boot[dataset][start:stop] = np.take_along_axis(
                draw_samples, draw_indices, axis=1
            ).mean(axis=1)

    macro_samples = np.mean(
        np.stack([boot[dataset] for dataset in args.datasets]), axis=0
    )
    macro_baseline = float(
        np.mean([observed[name]["baseline_wer_draw_mean"] for name in args.datasets])
    )
    macro_candidate = float(
        np.mean([observed[name]["candidate_wer_draw_mean"] for name in args.datasets])
    )
    macro_estimate = macro_candidate - macro_baseline
    result = {
        "inference": "crossed_calibration_draw_by_utterance_bootstrap",
        "draws": draw_count,
        "baseline_label": args.baseline_label,
        "candidate_label": args.candidate_label,
        "datasets": list(args.datasets),
        "reps": args.reps,
        "seed": args.seed,
        "observed": observed,
        "macro_delta_by_draw": [float(value) for value in draw_macro_deltas],
        "bootstrap": {
            dataset: summarize(boot[dataset], observed[dataset]["delta_draw_mean"])
            for dataset in args.datasets
        },
        "macro": {
            "baseline_wer_draw_mean": macro_baseline,
            "candidate_wer_draw_mean": macro_candidate,
            **summarize(macro_samples, macro_estimate),
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "crossed_multidraw_bootstrap.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    lines = [
        "# Crossed calibration-draw × utterance WER bootstrap",
        "",
        f"{draw_count} calibration draws; {args.reps:,} replicates; seed `{args.seed}`.",
        "Each replicate resamples calibration draws and evaluation utterances. "
        "Utterance indices are shared across draws, and draw indices are shared "
        "across domains.",
        "",
        f"| Scope | {args.baseline_label} WER % | {args.candidate_label} WER % | "
        "Δ pp [95% CI] | p |",
        "|---|---:|---:|---:|---:|",
    ]
    for dataset in args.datasets:
        row = observed[dataset]
        stat = result["bootstrap"][dataset]
        lines.append(
            f"| {dataset} | {100*row['baseline_wer_draw_mean']:.3f} | "
            f"{100*row['candidate_wer_draw_mean']:.3f} | "
            f"{100*stat['estimate']:+.3f} "
            f"[{100*stat['ci95_lower']:+.3f}, {100*stat['ci95_upper']:+.3f}] | "
            f"{stat['p_two_sided_centered']:.4f} |"
        )
    macro = result["macro"]
    lines.append(
        f"| **macro** | **{100*macro['baseline_wer_draw_mean']:.3f}** | "
        f"**{100*macro['candidate_wer_draw_mean']:.3f}** | "
        f"**{100*macro['estimate']:+.3f} "
        f"[{100*macro['ci95_lower']:+.3f}, {100*macro['ci95_upper']:+.3f}]** | "
        f"**{macro['p_two_sided_centered']:.4f}** |"
    )
    lines.extend(
        [
            "",
            "Macro Δ by calibration draw (pp): `"
            + "/".join(f"{100*value:+.3f}" for value in draw_macro_deltas)
            + "`.",
            "",
            "With few calibration draws this interval is intentionally wider "
            "than the fixed-draw utterance bootstrap and remains descriptive.",
            "",
        ]
    )
    (args.output_dir / "CROSSED_MULTIDRAW_REPORT.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
