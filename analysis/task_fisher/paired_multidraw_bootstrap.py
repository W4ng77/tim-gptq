#!/usr/bin/env python3
"""Paired WER bootstrap for a draw-averaged candidate effect."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from paired_run_bootstrap import DEFAULT_DATASETS, aligned_errors


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pair",
        nargs=3,
        action="append",
        metavar=("BASELINE", "CANDIDATE_HARD", "CANDIDATE_OTHER"),
        help="A draw with one baseline run and candidate split over two runs.",
    )
    parser.add_argument(
        "--simple-pair",
        nargs=2,
        action="append",
        metavar=("BASELINE", "CANDIDATE"),
        help="A draw whose baseline and candidate each occupy one run directory.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    parser.add_argument("--reps", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--baseline-label", default="Baseline")
    parser.add_argument("--candidate-label", default="Candidate")
    return parser.parse_args()


def locate(run_dirs, dataset: str, side: str):
    matches = [
        path
        for path in (run_dir / f"{dataset}.jsonl" for run_dir in run_dirs)
        if path.is_file()
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one {side} file for {dataset}: {matches}"
        )
    return matches[0]


def summarize(samples, estimate):
    centered = samples - estimate
    count = int(np.count_nonzero(np.abs(centered) >= abs(estimate)))
    return {
        "estimate": float(estimate),
        "ci95_lower": float(np.quantile(samples, 0.025)),
        "ci95_upper": float(np.quantile(samples, 0.975)),
        "p_two_sided_centered": (count + 1.0) / (len(samples) + 1.0),
    }


def main():
    args = parse_args()
    pairs = []
    for baseline, hard, other in args.pair or []:
        pairs.append(([Path(baseline)], [Path(hard), Path(other)]))
    for baseline, candidate in args.simple_pair or []:
        pairs.append(([Path(baseline)], [Path(candidate)]))
    if not pairs:
        raise ValueError("At least one --pair or --simple-pair is required")
    data = {dataset: [] for dataset in args.datasets}
    for baseline_dirs, candidate_dirs in pairs:
        for dataset in args.datasets:
            baseline = locate(baseline_dirs, dataset, "baseline")
            candidate = locate(candidate_dirs, dataset, "candidate")
            arrays = aligned_errors(
                baseline,
                candidate,
            )
            data[dataset].append(arrays)

    rng = np.random.default_rng(args.seed)
    boot = {
        dataset: np.empty(args.reps, dtype=np.float64)
        for dataset in args.datasets
    }
    observed = {}
    for dataset, draws in data.items():
        lengths0 = draws[0][0]
        for lengths, _, _ in draws[1:]:
            if not np.array_equal(lengths0, lengths):
                raise ValueError(f"Reference lengths differ across draws: {dataset}")
        draw_deltas = []
        draw_baselines = []
        draw_candidates = []
        for lengths, baseline_errors, candidate_errors in draws:
            denominator = lengths.sum()
            draw_baselines.append(float(baseline_errors.sum() / denominator))
            draw_candidates.append(float(candidate_errors.sum() / denominator))
            draw_deltas.append(draw_candidates[-1] - draw_baselines[-1])
        observed[dataset] = {
            "baseline_wer_draw_mean": float(np.mean(draw_baselines)),
            "candidate_wer_draw_mean": float(np.mean(draw_candidates)),
            "delta_draw_mean": float(np.mean(draw_deltas)),
        }

        n = len(lengths0)
        for start in range(0, args.reps, args.batch_size):
            stop = min(start + args.batch_size, args.reps)
            indices = rng.integers(
                0, n, size=(stop - start, n), dtype=np.int32
            )
            deltas = []
            for lengths, baseline_errors, candidate_errors in draws:
                denominator = lengths[indices].sum(axis=1)
                deltas.append(
                    (
                        candidate_errors[indices].sum(axis=1)
                        - baseline_errors[indices].sum(axis=1)
                    )
                    / denominator
                )
            boot[dataset][start:stop] = np.mean(np.stack(deltas), axis=0)

    macro_samples = np.mean(
        np.stack([boot[dataset] for dataset in args.datasets]), axis=0
    )
    macro_baseline = float(
        np.mean(
            [observed[name]["baseline_wer_draw_mean"] for name in args.datasets]
        )
    )
    macro_candidate = float(
        np.mean(
            [observed[name]["candidate_wer_draw_mean"] for name in args.datasets]
        )
    )
    result = {
        "draws": len(pairs),
        "baseline_label": args.baseline_label,
        "candidate_label": args.candidate_label,
        "datasets": list(args.datasets),
        "reps": args.reps,
        "seed": args.seed,
        "observed": observed,
        "bootstrap": {
            dataset: summarize(
                boot[dataset], observed[dataset]["delta_draw_mean"]
            )
            for dataset in args.datasets
        },
        "macro": {
            "baseline_wer_draw_mean": macro_baseline,
            "candidate_wer_draw_mean": macro_candidate,
            **summarize(macro_samples, macro_candidate - macro_baseline),
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "paired_multidraw_bootstrap.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    macro = result["macro"]
    lines = [
        "# Draw-averaged paired stratified WER bootstrap",
        "",
        f"{len(pairs)} calibration draws; {args.reps:,} paired utterance "
        f"replicates; seed `{args.seed}`.",
        "",
        f"| Scope | {args.baseline_label} draw-mean WER % | "
        f"{args.candidate_label} draw-mean WER % | Δ pp [95% CI] | p |",
        "|---|---:|---:|---:|---:|",
    ]
    for dataset in args.datasets:
        row = observed[dataset]
        stat = result["bootstrap"][dataset]
        lines.append(
            f"| {dataset} | {100*row['baseline_wer_draw_mean']:.3f} | "
            f"{100*row['candidate_wer_draw_mean']:.3f} | "
            f"{100*stat['estimate']:+.3f} "
            f"[{100*stat['ci95_lower']:+.3f}, "
            f"{100*stat['ci95_upper']:+.3f}] | "
            f"{stat['p_two_sided_centered']:.4f} |"
        )
    lines.append(
        f"| **macro** | **{100*macro['baseline_wer_draw_mean']:.3f}** | "
        f"**{100*macro['candidate_wer_draw_mean']:.3f}** | "
        f"**{100*macro['estimate']:+.3f} "
        f"[{100*macro['ci95_lower']:+.3f}, "
        f"{100*macro['ci95_upper']:+.3f}]** | "
        f"**{macro['p_two_sided_centered']:.4f}** |"
    )
    lines.extend(
        [
            "",
            "The same utterance indices are resampled across all calibration "
            f"draws within each dataset. Negative Δ favors "
            f"{args.candidate_label}.",
            "",
        ]
    )
    (args.output_dir / "PAIRED_MULTIDRAW_REPORT.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
