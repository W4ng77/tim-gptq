#!/usr/bin/env python3
"""Paired external WER bootstrap stratified by a manifest metadata field."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from rapidfuzz.distance import Levenshtein

from paired_external_bootstrap import normalize, read_rows, summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uniform", type=Path, required=True)
    parser.add_argument("--task-density", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--field", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reps", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260729)
    args = parser.parse_args()

    manifest_payload = json.loads(args.manifest.read_text())
    manifest_rows = manifest_payload["selected"]
    left = read_rows(args.uniform / f"{args.dataset}.jsonl")
    right = read_rows(args.task_density / f"{args.dataset}.jsonl")
    if len(manifest_rows) != len(left) or len(left) != len(right):
        raise ValueError("Manifest/prediction row count mismatch")
    groups = {}
    # ProfASR IDs repeat across voices. Preserve the frozen manifest order
    # instead of keying metadata by ID, and verify every aligned row.
    for manifest_row, a, b in zip(manifest_rows, left, right, strict=True):
        if (str(manifest_row["id"]) != str(a["example_id"]) or
                a["example_id"] != b["example_id"] or
                a["reference"] != b["reference"]):
            raise ValueError(f"Alignment mismatch: {a['example_id']}/{b['example_id']}")
        reference = normalize(a["reference"]).split()
        group = str(manifest_row.get("metadata", {}).get(args.field, "missing"))
        groups.setdefault(group, [[], [], []])
        groups[group][0].append(len(reference))
        groups[group][1].append(Levenshtein.distance(reference, normalize(a["prediction"]).split()))
        groups[group][2].append(Levenshtein.distance(reference, normalize(b["prediction"]).split()))

    rng = np.random.default_rng(args.seed)
    result = {}
    for group, arrays in sorted(groups.items()):
        lengths, errors_a, errors_b = (np.asarray(values, dtype=np.int64) for values in arrays)
        estimate = (errors_b.sum() - errors_a.sum()) / lengths.sum()
        samples = np.empty(args.reps)
        for start in range(0, args.reps, 256):
            stop = min(args.reps, start + 256)
            draw = rng.integers(0, len(lengths), size=(stop - start, len(lengths)), dtype=np.int32)
            samples[start:stop] = ((errors_b[draw].sum(axis=1) - errors_a[draw].sum(axis=1)) /
                                   lengths[draw].sum(axis=1))
        result[group] = {"examples": len(lengths),
                         "uniform_wer": float(errors_a.sum() / lengths.sum()),
                         "task_density_wer": float(errors_b.sum() / lengths.sum()),
                         **summarize(samples, estimate)}
    payload = {"dataset": args.dataset, "field": args.field, "groups": result,
               "reps": args.reps, "seed": args.seed}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / f"stratified_by_{args.field}.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n")
    lines = [f"# Paired WER by {args.field}", "",
             "Contrast is bounded task density minus uniform GPTQ.", "",
             f"| {args.field} | N | Uniform % | Task density % | Δ pp [95% CI] | p |",
             "|---|---:|---:|---:|---:|---:|"]
    for group, row in result.items():
        lines.append(f"| {group} | {row['examples']} | {100*row['uniform_wer']:.3f} | "
                     f"{100*row['task_density_wer']:.3f} | {100*row['estimate']:+.3f} "
                     f"[{100*row['ci95_lower']:+.3f}, {100*row['ci95_upper']:+.3f}] | "
                     f"{row['p_two_sided_centered']:.4f} |")
    (args.output_dir / f"STRATIFIED_BY_{args.field.upper()}.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
