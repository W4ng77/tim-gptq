#!/usr/bin/env python3
"""Crossed model-scale x target-distribution bootstrap for Qwen external panels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from crossed_external_bootstrap import load_pair
from paired_external_bootstrap import summarize


FLEURS = ("fleurs-en-us", "fleurs-de-de", "fleurs-fr-fr", "fleurs-es-419", "fleurs-pt-br")
PROF = ("profasr-no-prompt",)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for model in ("q06", "q17"):
        for target in ("fleurs", "prof"):
            parser.add_argument(f"--{model}-{target}-pair", nargs=2, action="append",
                                required=True, metavar=("UNIFORM", "TASK_DENSITY"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reps", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260801)
    args = parser.parse_args()

    pair_args = {
        (0, 0): args.q06_fleurs_pair,
        (0, 1): args.q06_prof_pair,
        (1, 0): args.q17_fleurs_pair,
        (1, 1): args.q17_prof_pair,
    }
    counts = {len(value) for value in pair_args.values()}
    if len(counts) != 1 or next(iter(counts)) < 2:
        raise ValueError("Every model-target cell needs the same >=2 calibration draws")
    draw_count = next(iter(counts))
    datasets = {0: FLEURS, 1: PROF}
    data = {}
    for (model, target), pairs in pair_args.items():
        data[(model, target)] = {
            dataset: [load_pair(Path(a), Path(b), dataset) for a, b in pairs]
            for dataset in datasets[target]
        }

    # Verify references are identical across model and draw within each dataset.
    for target, target_datasets in datasets.items():
        for dataset in target_datasets:
            reference = data[(0, target)][dataset][0][0]
            for model in (0, 1):
                for lengths, _, _ in data[(model, target)][dataset]:
                    if not np.array_equal(reference, lengths):
                        raise ValueError(f"Reference mismatch: {target}/{dataset}")

    effects = np.empty((2, 2, draw_count), dtype=np.float64)
    for model in (0, 1):
        for target in (0, 1):
            for draw in range(draw_count):
                values = []
                for dataset in datasets[target]:
                    lengths, uniform, task = data[(model, target)][dataset][draw]
                    values.append((task.sum() - uniform.sum()) / lengths.sum())
                effects[model, target, draw] = np.mean(values)

    # Interaction convention: (1.7B-0.6B on Prof) - (1.7B-0.6B on FLEURS).
    interaction_by_draw = (
        effects[1, 1] - effects[0, 1] - effects[1, 0] + effects[0, 0]
    )
    observed_interaction = float(interaction_by_draw.mean())
    observed = {
        "q06_fleurs": float(effects[0, 0].mean()),
        "q06_prof": float(effects[0, 1].mean()),
        "q17_fleurs": float(effects[1, 0].mean()),
        "q17_prof": float(effects[1, 1].mean()),
        "scale_effect_on_fleurs": float((effects[1, 0] - effects[0, 0]).mean()),
        "scale_effect_on_prof": float((effects[1, 1] - effects[0, 1]).mean()),
        "interaction": observed_interaction,
    }

    rng = np.random.default_rng(args.seed)
    interaction_samples = np.empty(args.reps, dtype=np.float64)
    scale_fleurs_samples = np.empty(args.reps, dtype=np.float64)
    scale_prof_samples = np.empty(args.reps, dtype=np.float64)
    for start in range(0, args.reps, 128):
        stop = min(args.reps, start + 128)
        size = stop - start
        sampled = np.zeros((size, 2, 2, draw_count), dtype=np.float64)
        for target in (0, 1):
            for dataset in datasets[target]:
                lengths = data[(0, target)][dataset][0][0]
                utterance = rng.integers(0, len(lengths), size=(size, len(lengths)),
                                         dtype=np.int32)
                denominator = lengths[utterance].sum(axis=1)
                for model in (0, 1):
                    for draw, (_, uniform, task) in enumerate(data[(model, target)][dataset]):
                        sampled[:, model, target, draw] += (
                            task[utterance].sum(axis=1) - uniform[utterance].sum(axis=1)
                        ) / denominator / len(datasets[target])
        draw_indices = rng.integers(0, draw_count, size=(size, draw_count), dtype=np.int32)
        crossed = np.empty((size, 2, 2), dtype=np.float64)
        for model in (0, 1):
            for target in (0, 1):
                crossed[:, model, target] = np.take_along_axis(
                    sampled[:, model, target, :], draw_indices, axis=1
                ).mean(axis=1)
        scale_fleurs_samples[start:stop] = crossed[:, 1, 0] - crossed[:, 0, 0]
        scale_prof_samples[start:stop] = crossed[:, 1, 1] - crossed[:, 0, 1]
        interaction_samples[start:stop] = (
            scale_prof_samples[start:stop] - scale_fleurs_samples[start:stop]
        )

    payload = {
        "inference": "crossed_shared_draw_and_target_utterance_bootstrap",
        "draws": draw_count,
        "interaction_convention": "(q17-q06 on ProfASR) - (q17-q06 on FLEURS)",
        "observed": observed,
        "interaction_by_draw": interaction_by_draw.tolist(),
        "scale_effect_on_fleurs": summarize(
            scale_fleurs_samples, observed["scale_effect_on_fleurs"]),
        "scale_effect_on_prof": summarize(
            scale_prof_samples, observed["scale_effect_on_prof"]),
        "interaction": summarize(interaction_samples, observed_interaction),
        "reps": args.reps,
        "seed": args.seed,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "crossed_scale_target_bootstrap.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n")
    stat = payload["interaction"]
    (args.output_dir / "CROSSED_SCALE_TARGET_REPORT.md").write_text(
        "# Crossed Qwen model-scale × target-distribution bootstrap\n\n"
        f"{draw_count} shared calibration draws; interaction convention: "
        "`(1.7B−0.6B on ProfASR) − (1.7B−0.6B on FLEURS)`.\n\n"
        f"Cell effects (task density minus uniform, pp): Qwen-0.6B FLEURS "
        f"`{100*observed['q06_fleurs']:+.3f}`, ProfASR "
        f"`{100*observed['q06_prof']:+.3f}`; Qwen-1.7B FLEURS "
        f"`{100*observed['q17_fleurs']:+.3f}`, ProfASR "
        f"`{100*observed['q17_prof']:+.3f}`.\n\n"
        f"Formal interaction: **{100*stat['estimate']:+.3f}pp "
        f"[{100*stat['ci95_lower']:+.3f}, {100*stat['ci95_upper']:+.3f}]**, "
        f"p=`{stat['p_two_sided_centered']:.4f}`. Draw interactions: `" +
        "/".join(f"{100*x:+.3f}" for x in interaction_by_draw) + "pp`.\n"
    )


if __name__ == "__main__":
    main()
