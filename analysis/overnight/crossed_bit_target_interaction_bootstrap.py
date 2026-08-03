#!/usr/bin/env python3
"""Shared-draw bootstrap for a bit-width x target x density interaction."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from crossed_external_bootstrap import load_pair
from paired_external_bootstrap import summarize


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cell", nargs=4, action="append", required=True,
        metavar=("BIT", "TARGET", "UNIFORM", "TASK"),
        help="Repeat once per draw for every bit x target cell.",
    )
    parser.add_argument(
        "--dataset", nargs=2, action="append", required=True,
        metavar=("TARGET", "DATASET"),
    )
    parser.add_argument("--bit-order", nargs=2, required=True)
    parser.add_argument("--target-order", nargs=2, required=True)
    parser.add_argument(
        "--factor-name",
        default="bit-width",
        help="Semantic name for the two-level factor; defaults to bit-width.",
    )
    parser.add_argument("--cap-errors-at-reference", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reps", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260801)
    args = parser.parse_args()

    bits = tuple(args.bit_order)
    targets = tuple(args.target_order)
    cells: dict[tuple[str, str], list[tuple[Path, Path]]] = defaultdict(list)
    for bit, target, uniform, task in args.cell:
        cells[(bit, target)].append((Path(uniform), Path(task)))
    expected = {(bit, target) for bit in bits for target in targets}
    if set(cells) != expected:
        raise ValueError(f"Expected cells {sorted(expected)}, got {sorted(cells)}")
    draw_counts = {len(cells[key]) for key in expected}
    if len(draw_counts) != 1 or next(iter(draw_counts)) < 2:
        raise ValueError("Every cell needs the same >=2 calibration draws")
    draw_count = next(iter(draw_counts))

    datasets: dict[str, list[str]] = defaultdict(list)
    for target, dataset in args.dataset:
        datasets[target].append(dataset)
    if set(datasets) != set(targets) or any(not datasets[t] for t in targets):
        raise ValueError("Each target in --target-order needs >=1 --dataset")

    data: dict[tuple[str, str, str], list[tuple[np.ndarray, np.ndarray, np.ndarray]]] = {}
    for bit in bits:
        for target in targets:
            for dataset in datasets[target]:
                data[(bit, target, dataset)] = [
                    load_pair(uniform, task, dataset, args.cap_errors_at_reference)
                    for uniform, task in cells[(bit, target)]
                ]

    effects = np.empty((2, 2, draw_count), dtype=np.float64)
    for bit_index, bit in enumerate(bits):
        for target_index, target in enumerate(targets):
            for draw in range(draw_count):
                per_dataset = []
                for dataset in datasets[target]:
                    lengths, uniform, task = data[(bit, target, dataset)][draw]
                    per_dataset.append((task.sum() - uniform.sum()) / lengths.sum())
                effects[bit_index, target_index, draw] = np.mean(per_dataset)

    target_by_draw = effects[:, 1, :] - effects[:, 0, :]
    interaction_by_draw = target_by_draw[1] - target_by_draw[0]
    observed_density = effects.mean(axis=2)
    observed_target = target_by_draw.mean(axis=1)
    observed_interaction = float(interaction_by_draw.mean())

    rng = np.random.default_rng(args.seed)
    density_samples = np.empty((2, 2, args.reps), dtype=np.float64)
    target_samples = np.empty((2, args.reps), dtype=np.float64)
    interaction_samples = np.empty(args.reps, dtype=np.float64)
    for start in range(0, args.reps, 128):
        stop = min(args.reps, start + 128)
        size = stop - start
        sampled = np.zeros((size, 2, 2, draw_count), dtype=np.float64)
        for target_index, target in enumerate(targets):
            for dataset in datasets[target]:
                reference = data[(bits[0], target, dataset)][0][0]
                for bit in bits:
                    for lengths, _, _ in data[(bit, target, dataset)]:
                        if not np.array_equal(reference, lengths):
                            raise ValueError(f"Reference mismatch: {bit}/{target}/{dataset}")
                utterance = rng.integers(
                    0, len(reference), size=(size, len(reference)), dtype=np.int32
                )
                denominator = reference[utterance].sum(axis=1)
                for bit_index, bit in enumerate(bits):
                    for draw, (_, uniform, task) in enumerate(data[(bit, target, dataset)]):
                        sampled[:, bit_index, target_index, draw] += (
                            task[utterance].sum(axis=1) - uniform[utterance].sum(axis=1)
                        ) / denominator / len(datasets[target])

        draw_indices = rng.integers(
            0, draw_count, size=(size, draw_count), dtype=np.int32
        )
        for bit_index in range(2):
            for target_index in range(2):
                density_samples[bit_index, target_index, start:stop] = (
                    np.take_along_axis(
                        sampled[:, bit_index, target_index, :], draw_indices, axis=1
                    ).mean(axis=1)
                )
            target_samples[bit_index, start:stop] = (
                density_samples[bit_index, 1, start:stop]
                - density_samples[bit_index, 0, start:stop]
            )
        interaction_samples[start:stop] = (
            target_samples[1, start:stop] - target_samples[0, start:stop]
        )

    density_payload = {
        bit: {
            target: summarize(
                density_samples[bit_index, target_index],
                effects[bit_index, target_index].mean(),
            )
            for target_index, target in enumerate(targets)
        }
        for bit_index, bit in enumerate(bits)
    }
    target_payload = {
        bit: summarize(target_samples[bit_index], observed_target[bit_index])
        for bit_index, bit in enumerate(bits)
    }
    payload = {
        "inference": "shared_draw_and_target_utterance_bootstrap",
        "draws": draw_count,
        "error_cap": "reference_length" if args.cap_errors_at_reference else None,
        "density_convention": "task_minus_uniform",
        "target_convention": f"{targets[1]} minus {targets[0]}",
        "bit_target_interaction_convention": (
            f"({targets[1]}−{targets[0]})@{bits[1]} minus "
            f"({targets[1]}−{targets[0]})@{bits[0]}"
        ),
        "factor_name": args.factor_name,
        "density_effects": density_payload,
        "density_effects_by_draw": {
            bit: {
                target: effects[bit_index, target_index].tolist()
                for target_index, target in enumerate(targets)
            }
            for bit_index, bit in enumerate(bits)
        },
        "target_interactions": target_payload,
        "target_interactions_by_draw": {
            bit: target_by_draw[bit_index].tolist()
            for bit_index, bit in enumerate(bits)
        },
        "bit_target_interaction": summarize(interaction_samples, observed_interaction),
        "bit_target_interaction_by_draw": interaction_by_draw.tolist(),
        "reps": args.reps,
        "seed": args.seed,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "crossed_bit_target_interaction.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    stat = payload["bit_target_interaction"]
    lines = [
        f"# Crossed {args.factor_name} × target × density interaction",
        "",
        f"{draw_count} shared calibration draws; density is task minus uniform; "
        f"target is `{targets[1]} − {targets[0]}`; {args.factor_name} contrast is "
        f"`{bits[1]} − {bits[0]}`; error cap: `{payload['error_cap']}`.",
        "",
        f"| {args.factor_name.capitalize()} | " + " | ".join(targets) + " | Target interaction |",
        "|---|" + "---:|" * (len(targets) + 1),
    ]
    for bit_index, bit in enumerate(bits):
        cells_text = [
            f"{100 * observed_density[bit_index, target_index]:+.3f}pp"
            for target_index in range(2)
        ]
        tstat = target_payload[bit]
        lines.append(
            f"| {bit} | " + " | ".join(cells_text) +
            f" | {100*tstat['estimate']:+.3f}pp "
            f"[{100*tstat['ci95_lower']:+.3f}, {100*tstat['ci95_upper']:+.3f}] |"
        )
    lines.extend([
        "",
        f"Three-way interaction: **{100*stat['estimate']:+.3f}pp "
        f"[{100*stat['ci95_lower']:+.3f}, {100*stat['ci95_upper']:+.3f}]**, "
        f"p=`{stat['p_two_sided_centered']:.4f}`. Draw interactions: `" +
        "/".join(f"{100*x:+.3f}" for x in interaction_by_draw) + "pp`.",
    ])
    (args.output_dir / "CROSSED_BIT_TARGET_INTERACTION_REPORT.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
