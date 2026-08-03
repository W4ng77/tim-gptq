#!/usr/bin/env python3
"""Shared-draw bootstrap for one model's target-distribution interaction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from crossed_external_bootstrap import load_pair
from paired_external_bootstrap import summarize


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-a-pair", nargs=2, action="append", required=True)
    parser.add_argument("--target-b-pair", nargs=2, action="append", required=True)
    parser.add_argument("--target-a-dataset", action="append", required=True)
    parser.add_argument("--target-b-dataset", action="append", required=True)
    parser.add_argument("--target-a-label", required=True)
    parser.add_argument("--target-b-label", required=True)
    parser.add_argument("--cap-errors-at-reference", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reps", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260801)
    args = parser.parse_args()
    counts = {len(args.target_a_pair), len(args.target_b_pair)}
    if len(counts) != 1 or next(iter(counts)) < 2:
        raise ValueError("Both targets need the same >=2 calibration draws")
    draw_count = next(iter(counts))
    datasets = (args.target_a_dataset, args.target_b_dataset)
    pair_args = (args.target_a_pair, args.target_b_pair)
    data = []
    for target in (0, 1):
        data.append({
            dataset: [
                load_pair(Path(a), Path(b), dataset, args.cap_errors_at_reference)
                for a, b in pair_args[target]
            ]
            for dataset in datasets[target]
        })

    effects = np.empty((2, draw_count), dtype=np.float64)
    for target in (0, 1):
        for draw in range(draw_count):
            values = []
            for dataset in datasets[target]:
                lengths, uniform, task = data[target][dataset][draw]
                values.append((task.sum() - uniform.sum()) / lengths.sum())
            effects[target, draw] = np.mean(values)
    interaction_by_draw = effects[1] - effects[0]
    target_effects = effects.mean(axis=1)
    observed_interaction = float(interaction_by_draw.mean())

    rng = np.random.default_rng(args.seed)
    interaction_samples = np.empty(args.reps, dtype=np.float64)
    target_samples = np.empty((2, args.reps), dtype=np.float64)
    for start in range(0, args.reps, 128):
        stop = min(args.reps, start + 128)
        size = stop - start
        sampled = np.zeros((size, 2, draw_count), dtype=np.float64)
        for target in (0, 1):
            for dataset in datasets[target]:
                lengths = data[target][dataset][0][0]
                for other, _, _ in data[target][dataset][1:]:
                    if not np.array_equal(lengths, other):
                        raise ValueError(f"Reference mismatch: target {target}/{dataset}")
                utterance = rng.integers(
                    0, len(lengths), size=(size, len(lengths)), dtype=np.int32
                )
                denominator = lengths[utterance].sum(axis=1)
                for draw, (_, uniform, task) in enumerate(data[target][dataset]):
                    sampled[:, target, draw] += (
                        task[utterance].sum(axis=1) - uniform[utterance].sum(axis=1)
                    ) / denominator / len(datasets[target])
        draw_indices = rng.integers(
            0, draw_count, size=(size, draw_count), dtype=np.int32
        )
        for target in (0, 1):
            target_samples[target, start:stop] = np.take_along_axis(
                sampled[:, target, :], draw_indices, axis=1
            ).mean(axis=1)
        interaction_samples[start:stop] = (
            target_samples[1, start:stop] - target_samples[0, start:stop]
        )

    labels = (args.target_a_label, args.target_b_label)
    payload = {
        "inference": "shared_calibration_draw_and_target_utterance_bootstrap",
        "draws": draw_count,
        "error_cap": "reference_length" if args.cap_errors_at_reference else None,
        "interaction_convention": f"{labels[1]} minus {labels[0]}",
        "target_effects": {
            labels[target]: summarize(target_samples[target], target_effects[target])
            for target in (0, 1)
        },
        "target_effects_by_draw": {
            labels[target]: effects[target].tolist() for target in (0, 1)
        },
        "interaction": summarize(interaction_samples, observed_interaction),
        "interaction_by_draw": interaction_by_draw.tolist(),
        "reps": args.reps,
        "seed": args.seed,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "crossed_target_interaction_bootstrap.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    stat = payload["interaction"]
    (args.output_dir / "CROSSED_TARGET_INTERACTION_REPORT.md").write_text(
        "# Crossed target-distribution interaction\n\n"
        f"{draw_count} shared calibration draws; contrast is "
        f"`{labels[1]} − {labels[0]}`; "
        f"error cap: `{payload['error_cap']}`.\n\n"
        f"Target effects (task density minus uniform, pp): "
        f"{labels[0]} `{100*target_effects[0]:+.3f}`, "
        f"{labels[1]} `{100*target_effects[1]:+.3f}`.\n\n"
        f"Interaction: **{100*stat['estimate']:+.3f}pp "
        f"[{100*stat['ci95_lower']:+.3f}, {100*stat['ci95_upper']:+.3f}]**, "
        f"p=`{stat['p_two_sided_centered']:.4f}`. Draw interactions: `" +
        "/".join(f"{100*x:+.3f}" for x in interaction_by_draw) + "pp`.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
