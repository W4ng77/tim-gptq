#!/usr/bin/env python3
"""Paired utterance bootstrap for two completed external ASR runs."""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path

import numpy as np
from rapidfuzz.distance import Levenshtein


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text)).lower().strip()
    text = "".join(ch for ch in text if not unicodedata.category(ch).startswith("P"))
    return re.sub(r"\s+", " ", text).strip()


def read_rows(path: Path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def summarize(samples, estimate):
    centered = samples - estimate
    p = (np.count_nonzero(np.abs(centered) >= abs(estimate)) + 1) / (len(samples) + 1)
    return {"estimate": float(estimate),
            "ci95_lower": float(np.quantile(samples, .025)),
            "ci95_upper": float(np.quantile(samples, .975)),
            "p_two_sided_centered": float(p)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uniform", type=Path, required=True)
    parser.add_argument("--task-density", type=Path, required=True)
    parser.add_argument("--dataset", action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reps", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260729)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    results = {}
    boot = []
    for dataset in args.dataset:
        left = read_rows(args.uniform / f"{dataset}.jsonl")
        right = read_rows(args.task_density / f"{dataset}.jsonl")
        if len(left) != len(right):
            raise ValueError(f"Row count mismatch for {dataset}")
        lengths, left_errors, right_errors = [], [], []
        for a, b in zip(left, right, strict=True):
            if a["example_id"] != b["example_id"] or a["reference"] != b["reference"]:
                raise ValueError(f"Alignment mismatch for {dataset}/{a['example_id']}")
            reference = normalize(a["reference"]).split()
            if not reference:
                raise ValueError(f"Empty normalized reference for {dataset}/{a['example_id']}")
            lengths.append(len(reference))
            left_errors.append(Levenshtein.distance(reference, normalize(a["prediction"]).split()))
            right_errors.append(Levenshtein.distance(reference, normalize(b["prediction"]).split()))
        lengths = np.asarray(lengths, dtype=np.int64)
        delta = np.asarray(right_errors, dtype=np.int64) - np.asarray(left_errors, dtype=np.int64)
        estimate = delta.sum() / lengths.sum()
        samples = np.empty(args.reps)
        for start in range(0, args.reps, 256):
            stop = min(args.reps, start + 256)
            draw = rng.integers(0, len(lengths), size=(stop - start, len(lengths)), dtype=np.int32)
            samples[start:stop] = delta[draw].sum(axis=1) / lengths[draw].sum(axis=1)
        results[dataset] = summarize(samples, estimate)
        boot.append(samples)
    macro_estimate = np.mean([value["estimate"] for value in results.values()])
    macro_samples = np.mean(np.stack(boot), axis=0)
    payload = {"contrast": "task_density_minus_uniform", "datasets": results,
               "macro": summarize(macro_samples, macro_estimate),
               "reps": args.reps, "seed": args.seed,
               "uniform": str(args.uniform.resolve()),
               "task_density": str(args.task_density.resolve())}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "paired_external_bootstrap.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n")
    lines = ["# Paired external-panel bootstrap", "",
             "Contrast is bounded task density minus uniform GPTQ; lower is better.", "",
             "| Dataset | Effect pp [95% CI] | p |", "|---|---:|---:|"]
    for dataset in (*args.dataset, "macro"):
        stat = payload["macro"] if dataset == "macro" else results[dataset]
        lines.append(f"| {dataset} | {100*stat['estimate']:+.3f} "
                     f"[{100*stat['ci95_lower']:+.3f}, {100*stat['ci95_upper']:+.3f}] | "
                     f"{stat['p_two_sided_centered']:.4f} |")
    (args.output_dir / "PAIRED_EXTERNAL_REPORT.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
