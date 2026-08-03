#!/usr/bin/env python3
"""Paired utterance bootstrap for template x support x density WER factorial."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from rapidfuzz.distance import Levenshtein

DATASETS = ("librispeech-other", "voxpopuli", "gigaspeech")
CELLS = tuple(f"{t}{s}{d}" for t in (0, 1) for s in (0, 1) for d in (0, 1))


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell", action="append", required=True, help="000=/run/path")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reps", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260729)
    return parser.parse_args()


def read_rows(path: Path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def parse_cells(values):
    cells = {}
    for value in values:
        key, raw_path = value.split("=", 1)
        if key not in CELLS or key in cells:
            raise ValueError(f"Invalid or repeated cell {key!r}")
        cells[key] = Path(raw_path)
    if set(cells) != set(CELLS):
        raise ValueError(f"Need cells {CELLS}, got {sorted(cells)}")
    return cells


def load_dataset(cells, dataset):
    rows = {key: read_rows(path / f"{dataset}.jsonl") for key, path in cells.items()}
    baseline = rows["000"]
    lengths = np.asarray([len(row["reference"].split()) for row in baseline], dtype=np.int64)
    errors = {}
    for key, values in rows.items():
        if len(values) != len(baseline):
            raise ValueError(f"Row count mismatch for {dataset}/{key}")
        current = []
        for left, right in zip(baseline, values, strict=True):
            if left["example_id"] != right["example_id"] or left["reference"] != right["reference"]:
                raise ValueError(f"Alignment mismatch for {dataset}/{key}")
            current.append(Levenshtein.distance(
                left["reference"].split(), right["prediction"].split()))
        errors[key] = np.asarray(current, dtype=np.int64)
    return lengths, errors


def coefficients():
    definitions = {
        "template": ((0,), 4),
        "support": ((1,), 4),
        "density": ((2,), 4),
        "template:support": ((0, 1), 2),
        "template:density": ((0, 2), 2),
        "support:density": ((1, 2), 2),
        "template:support:density": ((0, 1, 2), 1),
    }
    result = {}
    for name, (axes, divisor) in definitions.items():
        result[name] = {
            key: np.prod([1 if int(key[axis]) else -1 for axis in axes]) / divisor
            for key in CELLS
        }
    return result


def summarize(samples, estimate):
    centered = samples - estimate
    p = (np.count_nonzero(np.abs(centered) >= abs(estimate)) + 1) / (len(samples) + 1)
    return {"estimate": float(estimate), "ci95_lower": float(np.quantile(samples, .025)),
            "ci95_upper": float(np.quantile(samples, .975)),
            "p_two_sided_centered": float(p)}


def main():
    args = arguments()
    cells = parse_cells(args.cell)
    coefs = coefficients()
    rng = np.random.default_rng(args.seed)
    observed = {}
    boot = {name: [] for name in coefs}
    cell_wer = {}
    for dataset in DATASETS:
        lengths, errors = load_dataset(cells, dataset)
        denominator = int(lengths.sum())
        wers = {key: float(value.sum() / denominator) for key, value in errors.items()}
        cell_wer[dataset] = wers
        observed[dataset] = {
            name: float(sum(weights[key] * wers[key] for key in CELLS))
            for name, weights in coefs.items()
        }
        samples = {name: np.empty(args.reps) for name in coefs}
        for start in range(0, args.reps, 256):
            stop = min(args.reps, start + 256)
            draw = rng.integers(0, len(lengths), size=(stop - start, len(lengths)), dtype=np.int32)
            denom = lengths[draw].sum(axis=1)
            draw_wers = {key: values[draw].sum(axis=1) / denom for key, values in errors.items()}
            for name, weights in coefs.items():
                samples[name][start:stop] = sum(weights[key] * draw_wers[key] for key in CELLS)
        for name in coefs:
            boot[name].append(samples[name])
    macro_observed = {name: float(np.mean([observed[d][name] for d in DATASETS])) for name in coefs}
    macro = {name: summarize(np.mean(np.stack(boot[name]), axis=0), macro_observed[name])
             for name in coefs}
    result = {"factor_bits": {"0": "minimal/prompt/uniform", "1": "inference/full/KL"},
              "cells": {key: str(path.resolve()) for key, path in cells.items()},
              "cell_wer": cell_wer, "observed_contrasts": observed,
              "macro_contrasts": macro, "reps": args.reps, "seed": args.seed}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "three_factor_bootstrap.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n")
    lines = ["# Prompt template × row support × density", "",
             "Bits are template `{minimal,inference}`, support `{prompt,full}`, and density `{uniform,KL}`.", "",
             "| Contrast | Macro effect pp [95% CI] | p |", "|---|---:|---:|"]
    for name, stat in macro.items():
        lines.append(f"| {name} | {100*stat['estimate']:+.3f} [{100*stat['ci95_lower']:+.3f}, {100*stat['ci95_upper']:+.3f}] | {stat['p_two_sided_centered']:.4f} |")
    (args.output_dir / "THREE_FACTOR_REPORT.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
