#!/usr/bin/env python3
"""Crossed calibration-draw x utterance bootstrap for exact entity recall."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from paired_entity_bootstrap import contains_phrase
from paired_external_bootstrap import normalize, read_rows, summarize


def load_pair(uniform: Path, candidate: Path, manifest: dict, dataset: str):
    left = read_rows(uniform / f"{dataset}.jsonl")
    right = read_rows(candidate / f"{dataset}.jsonl")
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
    return tuple(np.asarray(v, dtype=np.int64) for v in (totals, left_hits, right_hits))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair", nargs=2, action="append", required=True,
                        metavar=("UNIFORM", "TASK_DENSITY"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reps", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument("--baseline-label", default="Uniform GPTQ")
    parser.add_argument("--candidate-label", default="Task density")
    args = parser.parse_args()
    if len(args.pair) < 2:
        raise ValueError("Need at least two calibration draws")

    manifest_payload = json.loads(args.manifest.read_text())
    manifest = {str(row["id"]): row for row in manifest_payload["selected"]}
    draws = [load_pair(Path(a), Path(b), manifest, args.dataset) for a, b in args.pair]
    totals = draws[0][0]
    for other, _, _ in draws[1:]:
        if not np.array_equal(totals, other):
            raise ValueError("Entity counts differ across calibration draws")
    if totals.sum() == 0:
        raise ValueError("No entities found")

    uniform = np.asarray([left.sum() / totals.sum() for _, left, _ in draws])
    task = np.asarray([right.sum() / totals.sum() for _, _, right in draws])
    effects = task - uniform
    estimate = effects.mean()
    samples = np.empty(args.reps)
    rng = np.random.default_rng(args.seed)
    for start in range(0, args.reps, 128):
        stop = min(args.reps, start + 128)
        size = stop - start
        utterance = rng.integers(0, len(totals), size=(size, len(totals)), dtype=np.int32)
        denominator = totals[utterance].sum(axis=1)
        draw_samples = np.empty((size, len(draws)))
        for index, (_, left, right) in enumerate(draws):
            draw_samples[:, index] = ((right[utterance].sum(axis=1) -
                                       left[utterance].sum(axis=1)) / denominator)
        draw_indices = rng.integers(0, len(draws), size=(size, len(draws)), dtype=np.int32)
        samples[start:stop] = np.take_along_axis(draw_samples, draw_indices, axis=1).mean(axis=1)

    payload = {
        "contrast": "candidate_minus_baseline_entity_exact_recall",
        "baseline_label": args.baseline_label,
        "candidate_label": args.candidate_label,
        "inference": "crossed_calibration_draw_by_utterance_bootstrap",
        "dataset": args.dataset,
        "draws": len(draws),
        "examples": len(totals),
        "entities": int(totals.sum()),
        "uniform_recall_draw_mean": float(uniform.mean()),
        "task_density_recall_draw_mean": float(task.mean()),
        "delta_by_draw": effects.tolist(),
        **summarize(samples, estimate),
        "reps": args.reps,
        "seed": args.seed,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "crossed_entity_bootstrap.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n")
    (args.output_dir / "CROSSED_ENTITY_REPORT.md").write_text(
        "# Crossed exact-entity-recall bootstrap\n\n"
        f"{len(draws)} calibration draws, {payload['entities']:,} entities over "
        f"{payload['examples']:,} utterances. {args.baseline_label}/"
        f"{args.candidate_label} draw-mean recall: "
        f"{100*payload['uniform_recall_draw_mean']:.3f}% / "
        f"{100*payload['task_density_recall_draw_mean']:.3f}%. Difference: "
        f"{100*payload['estimate']:+.3f}pp "
        f"[{100*payload['ci95_lower']:+.3f}, {100*payload['ci95_upper']:+.3f}], "
        f"p={payload['p_two_sided_centered']:.4f}. Draw effects: `" +
        "/".join(f"{100*x:+.3f}" for x in effects) + "pp`.\n"
    )


if __name__ == "__main__":
    main()
