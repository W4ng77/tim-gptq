#!/usr/bin/env python3
"""Nested calibration-draw x permutation-seed x utterance WER bootstrap.

Each ``--draw`` supplies one uniform run, one aligned task-measure run, and two
or more marginal-preserving permutation runs for the same calibration draw.
The analysis estimates both permutation-minus-aligned (correspondence) and
permutation-minus-uniform (raw baseline) contrasts without treating the
permutation replicates as independent calibration draws.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


TASK_FISHER_DIR = Path(__file__).resolve().parents[1] / "task_fisher"
sys.path.insert(0, str(TASK_FISHER_DIR))

from paired_multidraw_bootstrap import locate, summarize  # noqa: E402
from paired_run_bootstrap import DEFAULT_DATASETS, aligned_errors  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--draw",
        nargs="+",
        action="append",
        metavar="RUN",
        help=(
            "UNIFORM ALIGNED PERMUTED [PERMUTED ...] for one calibration draw; "
            "repeat once per draw"
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    parser.add_argument("--reps", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--batch-size", type=int, default=128)
    return parser.parse_args()


def validate_args(args):
    if not args.draw or len(args.draw) < 2:
        raise ValueError("Nested inference requires at least two calibration draws")
    widths = {len(draw) for draw in args.draw}
    if min(widths) < 4:
        raise ValueError("Each --draw needs uniform, aligned, and >=2 permutations")
    permutation_counts = {width - 2 for width in widths}
    if len(permutation_counts) != 1:
        raise ValueError("Every calibration draw must have the same permutation count")
    if args.reps < 100 or args.batch_size < 1:
        raise ValueError("reps must be >=100 and batch-size must be positive")


def load_draw(dataset: str, paths: list[str]):
    uniform_dir, aligned_dir, *permutation_dirs = map(Path, paths)
    uniform_path = locate([uniform_dir], dataset, "uniform")
    aligned_path = locate([aligned_dir], dataset, "aligned")
    lengths, uniform_errors, aligned_task_errors = aligned_errors(
        uniform_path, aligned_path
    )
    permutation_errors = []
    for permutation_dir in permutation_dirs:
        permutation_path = locate([permutation_dir], dataset, "permuted")
        perm_lengths, perm_uniform_errors, perm_errors = aligned_errors(
            uniform_path, permutation_path
        )
        if not np.array_equal(lengths, perm_lengths):
            raise ValueError(f"Reference lengths differ within draw: {dataset}")
        if not np.array_equal(uniform_errors, perm_uniform_errors):
            raise ValueError(f"Uniform predictions differ within draw: {dataset}")
        permutation_errors.append(perm_errors)
    return lengths, uniform_errors, aligned_task_errors, np.stack(permutation_errors)


def observed_dataset(draws):
    uniform_wers = []
    aligned_wers = []
    permutation_wers = []
    for lengths, uniform, aligned, permutations in draws:
        denominator = lengths.sum()
        uniform_wers.append(float(uniform.sum() / denominator))
        aligned_wers.append(float(aligned.sum() / denominator))
        permutation_wers.append(
            [float(errors.sum() / denominator) for errors in permutations]
        )
    uniform_wers = np.asarray(uniform_wers)
    aligned_wers = np.asarray(aligned_wers)
    permutation_wers = np.asarray(permutation_wers)
    permuted_mean = float(permutation_wers.mean())
    aligned_mean = float(aligned_wers.mean())
    uniform_mean = float(uniform_wers.mean())
    return {
        "uniform_wer_draw_mean": uniform_mean,
        "aligned_wer_draw_mean": aligned_mean,
        "permuted_wer_nested_mean": permuted_mean,
        "permuted_wer_by_draw_seed": permutation_wers.tolist(),
        "permuted_minus_aligned": permuted_mean - aligned_mean,
        "permuted_minus_uniform": permuted_mean - uniform_mean,
    }


def main():
    args = parse_args()
    validate_args(args)
    draw_count = len(args.draw)
    permutation_count = len(args.draw[0]) - 2

    data = {
        dataset: [load_draw(dataset, paths) for paths in args.draw]
        for dataset in args.datasets
    }
    for dataset, draws in data.items():
        lengths0 = draws[0][0]
        for lengths, _, _, _ in draws[1:]:
            if not np.array_equal(lengths0, lengths):
                raise ValueError(f"Reference lengths differ across draws: {dataset}")

    observed = {
        dataset: observed_dataset(draws) for dataset, draws in data.items()
    }
    rng = np.random.default_rng(args.seed)
    boot = {
        contrast: {
            dataset: np.empty(args.reps, dtype=np.float64)
            for dataset in args.datasets
        }
        for contrast in ("permuted_minus_aligned", "permuted_minus_uniform")
    }

    for start in range(0, args.reps, args.batch_size):
        stop = min(start + args.batch_size, args.reps)
        size = stop - start
        # Calibration and permutation identities describe shared quantized
        # artifacts, so their resampling indices are shared across domains.
        draw_indices = rng.integers(
            0, draw_count, size=(size, draw_count), dtype=np.int32
        )
        permutation_indices = rng.integers(
            0,
            permutation_count,
            size=(size, draw_count, permutation_count),
            dtype=np.int32,
        )
        for dataset, draws in data.items():
            lengths = draws[0][0]
            utterance_indices = rng.integers(
                0, len(lengths), size=(size, len(lengths)), dtype=np.int32
            )
            denominator = lengths[utterance_indices].sum(axis=1)
            delta_aligned = np.empty(
                (size, draw_count, permutation_count), dtype=np.float64
            )
            delta_uniform = np.empty_like(delta_aligned)
            for draw_index, (_, uniform, aligned, permutations) in enumerate(draws):
                aligned_sum = aligned[utterance_indices].sum(axis=1)
                uniform_sum = uniform[utterance_indices].sum(axis=1)
                for permutation_index, permutation in enumerate(permutations):
                    permutation_sum = permutation[utterance_indices].sum(axis=1)
                    delta_aligned[:, draw_index, permutation_index] = (
                        permutation_sum - aligned_sum
                    ) / denominator
                    delta_uniform[:, draw_index, permutation_index] = (
                        permutation_sum - uniform_sum
                    ) / denominator

            for contrast, delta in (
                ("permuted_minus_aligned", delta_aligned),
                ("permuted_minus_uniform", delta_uniform),
            ):
                selected_draws = np.take_along_axis(
                    delta, draw_indices[:, :, None], axis=1
                )
                selected_permutations = np.take_along_axis(
                    selected_draws, permutation_indices, axis=2
                )
                boot[contrast][dataset][start:stop] = selected_permutations.mean(
                    axis=(1, 2)
                )

    result = {
        "inference": (
            "nested_calibration_draw_by_permutation_seed_by_utterance_bootstrap"
        ),
        "draws": draw_count,
        "permutation_seeds_per_draw": permutation_count,
        "datasets": list(args.datasets),
        "reps": args.reps,
        "seed": args.seed,
        "observed": observed,
        "bootstrap": {},
    }
    for contrast in boot:
        result["bootstrap"][contrast] = {
            dataset: summarize(boot[contrast][dataset], observed[dataset][contrast])
            for dataset in args.datasets
        }
        macro_samples = np.mean(
            np.stack([boot[contrast][dataset] for dataset in args.datasets]), axis=0
        )
        macro_estimate = float(
            np.mean([observed[dataset][contrast] for dataset in args.datasets])
        )
        result["bootstrap"][contrast]["macro"] = summarize(
            macro_samples, macro_estimate
        )

    result["macro_wer"] = {
        key: float(np.mean([observed[dataset][key] for dataset in args.datasets]))
        for key in (
            "uniform_wer_draw_mean",
            "aligned_wer_draw_mean",
            "permuted_wer_nested_mean",
        )
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "nested_permutation_bootstrap.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    lines = [
        "# Nested permutation identification bootstrap",
        "",
        f"{draw_count} calibration draws x {permutation_count} permutation seeds; "
        f"{args.reps:,} replicates; seed `{args.seed}`.",
        "Calibration draws, permutation seeds nested within draw, and evaluation "
        "utterances are resampled hierarchically. Artifact indices are shared "
        "across domains; utterance resampling is domain-specific.",
        "",
    ]
    for contrast, label in (
        ("permuted_minus_aligned", "Permuted - aligned"),
        ("permuted_minus_uniform", "Permuted - uniform"),
    ):
        lines.extend(
            [
                f"## {label}",
                "",
                "| Scope | Delta pp [95% CI] | p |",
                "|---|---:|---:|",
            ]
        )
        for dataset in args.datasets:
            stat = result["bootstrap"][contrast][dataset]
            lines.append(
                f"| {dataset} | {100 * stat['estimate']:+.3f} "
                f"[{100 * stat['ci95_lower']:+.3f}, "
                f"{100 * stat['ci95_upper']:+.3f}] | "
                f"{stat['p_two_sided_centered']:.4f} |"
            )
        macro = result["bootstrap"][contrast]["macro"]
        lines.extend(
            [
                f"| **macro** | **{100 * macro['estimate']:+.3f} "
                f"[{100 * macro['ci95_lower']:+.3f}, "
                f"{100 * macro['ci95_upper']:+.3f}]** | "
                f"**{macro['p_two_sided_centered']:.4f}** |",
                "",
            ]
        )
    lines.extend(
        [
            "Positive Permuted - aligned means breaking weight-state "
            "correspondence hurts. A null Permuted - uniform contrast means "
            "the preserved weight marginal alone does not beat uniform GPTQ.",
            "",
        ]
    )
    (args.output_dir / "NESTED_PERMUTATION_REPORT.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
