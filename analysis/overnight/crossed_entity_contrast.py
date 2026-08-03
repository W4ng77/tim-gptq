#!/usr/bin/env python3
"""Named crossed-draw bootstrap contrast for exact entity recall."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from paired_entity_bootstrap import contains_phrase
from paired_external_bootstrap import normalize, read_rows, summarize


def load_pair(baseline: Path, candidate: Path, manifest: dict, dataset: str):
    left = read_rows(baseline / f"{dataset}.jsonl")
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
                        metavar=("BASELINE", "CANDIDATE"))
    parser.add_argument("--baseline-label", required=True)
    parser.add_argument("--candidate-label", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reps", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260801)
    args = parser.parse_args()

    manifest_payload = json.loads(args.manifest.read_text())
    manifest = {str(row["id"]): row for row in manifest_payload["selected"]}
    draws = [load_pair(Path(a), Path(b), manifest, args.dataset) for a, b in args.pair]
    totals = draws[0][0]
    for other, _, _ in draws[1:]:
        if not np.array_equal(totals, other):
            raise ValueError("Entity counts differ across calibration draws")
    if totals.sum() == 0:
        raise ValueError("No entities found")

    baseline = np.asarray([left.sum() / totals.sum() for _, left, _ in draws])
    candidate = np.asarray([right.sum() / totals.sum() for _, _, right in draws])
    effects = candidate - baseline
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
            draw_samples[:, index] = (
                right[utterance].sum(axis=1) - left[utterance].sum(axis=1)
            ) / denominator
        draw_indices = rng.integers(0, len(draws), size=(size, len(draws)), dtype=np.int32)
        samples[start:stop] = np.take_along_axis(
            draw_samples, draw_indices, axis=1
        ).mean(axis=1)

    payload = {
        "contrast": f"{args.candidate_label}_minus_{args.baseline_label}_entity_exact_recall",
        "inference": "crossed_calibration_draw_by_utterance_bootstrap",
        "dataset": args.dataset,
        "draws": len(draws),
        "examples": len(totals),
        "entities": int(totals.sum()),
        "baseline_label": args.baseline_label,
        "candidate_label": args.candidate_label,
        "baseline_recall_draw_mean": float(baseline.mean()),
        "candidate_recall_draw_mean": float(candidate.mean()),
        "delta_by_draw": effects.tolist(),
        **summarize(samples, estimate),
        "reps": args.reps,
        "seed": args.seed,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "crossed_entity_contrast.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    (args.output_dir / "CROSSED_ENTITY_CONTRAST.md").write_text(
        "# Named crossed exact-entity-recall contrast\n\n"
        f"Candidate minus baseline: **{args.candidate_label} − {args.baseline_label}**. "
        "Positive recall difference favors the candidate.\n\n"
        f"{len(draws)} calibration draw(s); {payload['entities']:,} entities over "
        f"{payload['examples']:,} utterances. Baseline/candidate draw-mean recall: "
        f"{100*payload['baseline_recall_draw_mean']:.3f}% / "
        f"{100*payload['candidate_recall_draw_mean']:.3f}%. Difference: "
        f"{100*payload['estimate']:+.3f}pp "
        f"[{100*payload['ci95_lower']:+.3f}, {100*payload['ci95_upper']:+.3f}], "
        f"p={payload['p_two_sided_centered']:.4f}. Draw effects: `"
        + "/".join(f"{100*x:+.3f}" for x in effects) + "pp`.\n"
    )


if __name__ == "__main__":
    main()
