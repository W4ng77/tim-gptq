#!/usr/bin/env python3
"""Crossed calibration-draw x utterance WER bootstrap by metadata stratum."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from rapidfuzz.distance import Levenshtein

from paired_external_bootstrap import normalize, read_rows, summarize


def load_draw(uniform: Path, candidate: Path, dataset: str, manifest_rows, field: str):
    left = read_rows(uniform / f"{dataset}.jsonl")
    right = read_rows(candidate / f"{dataset}.jsonl")
    if not (len(manifest_rows) == len(left) == len(right)):
        raise ValueError("Manifest/prediction row count mismatch")
    groups: dict[str, list[list[int]]] = {}
    ids = []
    for manifest_row, a, b in zip(manifest_rows, left, right, strict=True):
        if (str(manifest_row["id"]) != str(a["example_id"])
                or a["example_id"] != b["example_id"]
                or a["reference"] != b["reference"]):
            raise ValueError(f"Alignment mismatch: {a['example_id']}/{b['example_id']}")
        reference = normalize(a["reference"]).split()
        group = str(manifest_row.get("metadata", {}).get(field, "missing"))
        groups.setdefault(group, [[], [], []])
        groups[group][0].append(len(reference))
        groups[group][1].append(
            Levenshtein.distance(reference, normalize(a["prediction"]).split())
        )
        groups[group][2].append(
            Levenshtein.distance(reference, normalize(b["prediction"]).split())
        )
        ids.append(str(a["example_id"]))
    return ids, {
        group: tuple(np.asarray(values, dtype=np.int64) for values in arrays)
        for group, arrays in groups.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair", nargs=2, action="append", required=True,
                        metavar=("UNIFORM", "TASK_DENSITY"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--field", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reps", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260801)
    args = parser.parse_args()
    if len(args.pair) < 2:
        raise ValueError("Need at least two calibration draws")
    manifest_rows = json.loads(args.manifest.read_text())["selected"]
    draws = [
        load_draw(Path(a), Path(b), args.dataset, manifest_rows, args.field)
        for a, b in args.pair
    ]
    reference_ids = draws[0][0]
    if any(ids != reference_ids for ids, _ in draws[1:]):
        raise ValueError("Example IDs differ across calibration draws")
    group_names = sorted(draws[0][1])
    if any(sorted(groups) != group_names for _, groups in draws[1:]):
        raise ValueError("Metadata groups differ across calibration draws")

    rng = np.random.default_rng(args.seed)
    shared_draw_indices = rng.integers(
        0, len(draws), size=(args.reps, len(draws)), dtype=np.int32
    )
    result = {}
    for group in group_names:
        arrays = [groups[group] for _, groups in draws]
        base_lengths = arrays[0][0]
        if any(not np.array_equal(base_lengths, lengths) for lengths, _, _ in arrays[1:]):
            raise ValueError(f"Reference lengths differ across draws: {group}")
        uniform = np.asarray([errors.sum() / lengths.sum() for lengths, errors, _ in arrays])
        task = np.asarray([errors.sum() / lengths.sum() for lengths, _, errors in arrays])
        deltas = task - uniform
        estimate = float(deltas.mean())
        samples = np.empty(args.reps, dtype=np.float64)
        for start in range(0, args.reps, 256):
            stop = min(args.reps, start + 256)
            size = stop - start
            utterance = rng.integers(
                0, len(base_lengths), size=(size, len(base_lengths)), dtype=np.int32
            )
            denominator = base_lengths[utterance].sum(axis=1)
            draw_samples = np.empty((size, len(arrays)), dtype=np.float64)
            for index, (_, errors_u, errors_k) in enumerate(arrays):
                draw_samples[:, index] = (
                    errors_k[utterance].sum(axis=1) - errors_u[utterance].sum(axis=1)
                ) / denominator
            samples[start:stop] = np.take_along_axis(
                draw_samples, shared_draw_indices[start:stop], axis=1
            ).mean(axis=1)
        result[group] = {
            "examples": len(base_lengths),
            "uniform_wer_draw_mean": float(uniform.mean()),
            "task_density_wer_draw_mean": float(task.mean()),
            "delta_by_draw": deltas.tolist(),
            **summarize(samples, estimate),
        }

    payload = {
        "inference": "crossed_shared_calibration_draw_by_utterance_bootstrap",
        "dataset": args.dataset,
        "field": args.field,
        "draws": len(draws),
        "groups": result,
        "reps": args.reps,
        "seed": args.seed,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / f"crossed_stratified_by_{args.field}.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = [
        f"# Crossed WER by {args.field}", "",
        f"{len(draws)} shared calibration draws; task density minus uniform GPTQ.", "",
        f"| {args.field} | N | Uniform % | Task density % | Δ pp [95% CI] | Effects by draw |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for group, row in result.items():
        effects = "/".join(f"{100*x:+.3f}" for x in row["delta_by_draw"])
        lines.append(
            f"| {group} | {row['examples']} | {100*row['uniform_wer_draw_mean']:.3f} | "
            f"{100*row['task_density_wer_draw_mean']:.3f} | {100*row['estimate']:+.3f} "
            f"[{100*row['ci95_lower']:+.3f}, {100*row['ci95_upper']:+.3f}] | {effects} |"
        )
    (args.output_dir / f"CROSSED_STRATIFIED_BY_{args.field.upper()}.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
