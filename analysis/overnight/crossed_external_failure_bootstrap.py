#!/usr/bin/env python3
"""Crossed draw x utterance bootstrap for catastrophic ASR failure rates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from rapidfuzz.distance import Levenshtein

from paired_external_bootstrap import normalize, read_rows, summarize


def load_pair(uniform: Path, candidate: Path, dataset: str):
    left = read_rows(uniform / f"{dataset}.jsonl")
    right = read_rows(candidate / f"{dataset}.jsonl")
    ids, metrics = [], {"wer_over_100": [[], []], "length_over_3x": [[], []]}
    for a, b in zip(left, right, strict=True):
        if a["example_id"] != b["example_id"] or a["reference"] != b["reference"]:
            raise ValueError(f"Alignment mismatch for {dataset}/{a['example_id']}")
        reference = normalize(a["reference"]).split()
        if not reference:
            raise ValueError(f"Empty normalized reference for {dataset}/{a['example_id']}")
        predictions = (normalize(a["prediction"]).split(), normalize(b["prediction"]).split())
        ids.append(str(a["example_id"]))
        for arm, prediction in enumerate(predictions):
            errors = Levenshtein.distance(reference, prediction)
            metrics["wer_over_100"][arm].append(errors > len(reference))
            metrics["length_over_3x"][arm].append(len(prediction) > 3 * len(reference))
    return ids, {
        metric: tuple(np.asarray(arm, dtype=np.int8) for arm in arms)
        for metric, arms in metrics.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair", nargs=2, action="append", required=True,
                        metavar=("UNIFORM", "TASK_DENSITY"))
    parser.add_argument("--dataset", action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reps", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260801)
    args = parser.parse_args()
    if len(args.pair) < 2:
        raise ValueError("Need at least two calibration draws")
    pairs = [(Path(a), Path(b)) for a, b in args.pair]
    data = {
        dataset: [load_pair(a, b, dataset) for a, b in pairs]
        for dataset in args.dataset
    }
    for dataset, draws in data.items():
        for ids, _ in draws[1:]:
            if ids != draws[0][0]:
                raise ValueError(f"Example IDs differ across draws: {dataset}")

    rng = np.random.default_rng(args.seed)
    shared_draw_indices = rng.integers(
        0, len(pairs), size=(args.reps, len(pairs)), dtype=np.int32
    )
    result: dict[str, object] = {
        "inference": "crossed_shared_calibration_draw_by_utterance_bootstrap",
        "draws": len(pairs),
        "datasets": args.dataset,
        "metrics": {},
        "reps": args.reps,
        "seed": args.seed,
    }
    for metric in ("wer_over_100", "length_over_3x"):
        observed, boot = {}, {}
        macro_by_draw = np.zeros(len(pairs), dtype=np.float64)
        for dataset, draws in data.items():
            uniform = np.asarray([arms[metric][0].mean() for _, arms in draws])
            task = np.asarray([arms[metric][1].mean() for _, arms in draws])
            deltas = task - uniform
            observed[dataset] = {
                "n": len(draws[0][0]),
                "uniform_rate_draw_mean": float(uniform.mean()),
                "task_density_rate_draw_mean": float(task.mean()),
                "delta_draw_mean": float(deltas.mean()),
                "uniform_count_by_draw": [int(arms[metric][0].sum()) for _, arms in draws],
                "task_density_count_by_draw": [int(arms[metric][1].sum()) for _, arms in draws],
                "delta_by_draw": deltas.tolist(),
            }
            macro_by_draw += deltas / len(args.dataset)
            samples = np.empty(args.reps, dtype=np.float64)
            n = len(draws[0][0])
            for start in range(0, args.reps, 256):
                stop = min(args.reps, start + 256)
                size = stop - start
                utterance = rng.integers(0, n, size=(size, n), dtype=np.int32)
                draw_samples = np.empty((size, len(draws)), dtype=np.float64)
                for index, (_, arms) in enumerate(draws):
                    draw_samples[:, index] = (
                        arms[metric][1][utterance].mean(axis=1)
                        - arms[metric][0][utterance].mean(axis=1)
                    )
                samples[start:stop] = np.take_along_axis(
                    draw_samples, shared_draw_indices[start:stop], axis=1
                ).mean(axis=1)
            boot[dataset] = samples
        macro_uniform = float(np.mean([
            row["uniform_rate_draw_mean"] for row in observed.values()
        ]))
        macro_task = float(np.mean([
            row["task_density_rate_draw_mean"] for row in observed.values()
        ]))
        macro_estimate = macro_task - macro_uniform
        macro_samples = np.mean(np.stack([boot[d] for d in args.dataset]), axis=0)
        result["metrics"][metric] = {
            "observed": observed,
            "bootstrap": {
                dataset: summarize(boot[dataset], observed[dataset]["delta_draw_mean"])
                for dataset in args.dataset
            },
            "macro": {
                "uniform_rate_draw_mean": macro_uniform,
                "task_density_rate_draw_mean": macro_task,
                **summarize(macro_samples, macro_estimate),
            },
            "macro_delta_by_draw": macro_by_draw.tolist(),
        }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "crossed_external_failure_bootstrap.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = [
        "# Crossed catastrophic-failure bootstrap",
        "",
        f"{len(pairs)} calibration draws; task density minus uniform GPTQ. "
        "Calibration draws are shared across datasets.",
    ]
    for metric, title in (
        ("wer_over_100", "Utterance WER above 100%"),
        ("length_over_3x", "Prediction longer than 3× reference"),
    ):
        block = result["metrics"][metric]
        lines.extend([
            "", f"## {title}", "",
            "| Scope | Uniform rate | Task-density rate | Δ pp [95% CI] | Counts U/K by draw |",
            "|---|---:|---:|---:|---:|",
        ])
        for dataset in args.dataset:
            row = block["observed"][dataset]
            stat = block["bootstrap"][dataset]
            counts = "/".join(
                f"{u}:{k}" for u, k in zip(
                    row["uniform_count_by_draw"], row["task_density_count_by_draw"], strict=True
                )
            )
            lines.append(
                f"| {dataset} | {100*row['uniform_rate_draw_mean']:.3f}% | "
                f"{100*row['task_density_rate_draw_mean']:.3f}% | "
                f"{100*stat['estimate']:+.3f} [{100*stat['ci95_lower']:+.3f}, "
                f"{100*stat['ci95_upper']:+.3f}] | {counts} |"
            )
        stat = block["macro"]
        lines.append(
            f"| **macro** | **{100*stat['uniform_rate_draw_mean']:.3f}%** | "
            f"**{100*stat['task_density_rate_draw_mean']:.3f}%** | "
            f"**{100*stat['estimate']:+.3f} [{100*stat['ci95_lower']:+.3f}, "
            f"{100*stat['ci95_upper']:+.3f}]** | — |"
        )
    (args.output_dir / "CROSSED_EXTERNAL_FAILURE_REPORT.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
