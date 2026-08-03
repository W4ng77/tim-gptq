#!/usr/bin/env python3
"""Paired utterance bootstrap for exact entity recall on ContextASR."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from paired_external_bootstrap import normalize, read_rows, summarize


def contains_phrase(tokens, phrase):
    width = len(phrase)
    return width > 0 and any(tokens[i:i + width] == phrase
                             for i in range(len(tokens) - width + 1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uniform", type=Path, required=True)
    parser.add_argument("--task-density", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reps", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260729)
    args = parser.parse_args()

    manifest_payload = json.loads(args.manifest.read_text())
    manifest = {str(row["id"]): row for row in manifest_payload["selected"]}
    left = read_rows(args.uniform / f"{args.dataset}.jsonl")
    right = read_rows(args.task_density / f"{args.dataset}.jsonl")
    if len(left) != len(right):
        raise ValueError("Row count mismatch")
    totals, left_hits, right_hits = [], [], []
    for a, b in zip(left, right, strict=True):
        if a["example_id"] != b["example_id"]:
            raise ValueError(f"Alignment mismatch: {a['example_id']}/{b['example_id']}")
        entities = manifest[str(a["example_id"])].get("metadata", {}).get("entity_list", [])
        phrases = [normalize(entity).split() for entity in entities]
        phrases = [phrase for phrase in phrases if phrase]
        pred_a = normalize(a["prediction"]).split()
        pred_b = normalize(b["prediction"]).split()
        totals.append(len(phrases))
        left_hits.append(sum(contains_phrase(pred_a, phrase) for phrase in phrases))
        right_hits.append(sum(contains_phrase(pred_b, phrase) for phrase in phrases))
    totals = np.asarray(totals, dtype=np.int64)
    left_hits = np.asarray(left_hits, dtype=np.int64)
    right_hits = np.asarray(right_hits, dtype=np.int64)
    if totals.sum() == 0:
        raise ValueError("No entities found")
    estimate = (right_hits.sum() - left_hits.sum()) / totals.sum()
    samples = np.empty(args.reps)
    rng = np.random.default_rng(args.seed)
    for start in range(0, args.reps, 256):
        stop = min(args.reps, start + 256)
        draw = rng.integers(0, len(totals), size=(stop - start, len(totals)), dtype=np.int32)
        samples[start:stop] = ((right_hits[draw].sum(axis=1) - left_hits[draw].sum(axis=1)) /
                               totals[draw].sum(axis=1))
    payload = {"contrast": "task_density_minus_uniform_entity_exact_recall",
               "dataset": args.dataset, "examples": len(totals),
               "entities": int(totals.sum()),
               "uniform_recall": float(left_hits.sum() / totals.sum()),
               "task_density_recall": float(right_hits.sum() / totals.sum()),
               **summarize(samples, estimate), "reps": args.reps, "seed": args.seed}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "paired_entity_bootstrap.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n")
    (args.output_dir / "PAIRED_ENTITY_REPORT.md").write_text(
        "# Paired exact-entity-recall bootstrap\n\n"
        f"{payload['entities']:,} entities over {payload['examples']:,} utterances. "
        f"Uniform/task-density recall: {100*payload['uniform_recall']:.3f}% / "
        f"{100*payload['task_density_recall']:.3f}%. Difference: "
        f"{100*payload['estimate']:+.3f}pp "
        f"[{100*payload['ci95_lower']:+.3f}, {100*payload['ci95_upper']:+.3f}], "
        f"p={payload['p_two_sided_centered']:.4f}.\n")


if __name__ == "__main__":
    main()
