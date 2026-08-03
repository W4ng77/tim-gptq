#!/usr/bin/env python3
"""Recompute external-panel WER with Unicode punctuation normalization."""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path

from rapidfuzz.distance import Levenshtein


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text)).lower().strip()
    text = "".join(ch for ch in text if not unicodedata.category(ch).startswith("P"))
    return re.sub(r"\s+", " ", text).strip()


def parse_run(value: str):
    label, raw_path = value.split("=", 1)
    return label, Path(raw_path)


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def contains_phrase(tokens, phrase):
    width = len(phrase)
    return width > 0 and any(tokens[i : i + width] == phrase for i in range(len(tokens) - width + 1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, help="label=/run/path")
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifests = {}
    for path in args.manifest_dir.glob("*.json"):
        payload = json.loads(path.read_text())
        manifests[payload["dataset_alias"]] = {
            str(row["id"]): row for row in payload["selected"]
        }
    results = {}
    for label, run_dir in map(parse_run, args.run):
        if json.loads((run_dir / "status.json").read_text()).get("state") != "completed":
            raise ValueError(f"Run is not completed: {run_dir}")
        by_dataset = {}
        for path in sorted(run_dir.glob("*.jsonl")):
            dataset = path.stem
            rows = read_jsonl(path)
            total_words = 0
            total_errors = 0
            entity_hits = 0
            entity_total = 0
            for row in rows:
                reference = normalize(row["reference"]).split()
                prediction = normalize(row["prediction"]).split()
                if not reference:
                    raise ValueError(f"Empty normalized reference: {dataset}/{row['example_id']}")
                total_words += len(reference)
                total_errors += Levenshtein.distance(reference, prediction)
                manifest_row = manifests.get(dataset, {}).get(str(row["example_id"]), {})
                entities = manifest_row.get("metadata", {}).get("entity_list", [])
                for entity in entities:
                    phrase = normalize(entity).split()
                    if phrase:
                        entity_total += 1
                        entity_hits += int(contains_phrase(prediction, phrase))
            by_dataset[dataset] = {
                "examples": len(rows),
                "reference_words": total_words,
                "wer": total_errors / total_words,
                "entity_exact_recall": None if entity_total == 0 else entity_hits / entity_total,
                "entities": entity_total,
            }
        results[label] = {"run_dir": str(run_dir.resolve()), "datasets": by_dataset}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "external_metrics.json").write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n")
    lines = ["# External evaluation summary", "",
             "Unicode-NFKC, lowercase, Unicode-punctuation deletion, whitespace-token WER.", "",
             "| Run | Dataset | N | WER % | Entity exact recall % |", "|---|---|---:|---:|---:|"]
    for label, payload in results.items():
        for dataset, stat in payload["datasets"].items():
            recall = "—" if stat["entity_exact_recall"] is None else f"{100*stat['entity_exact_recall']:.2f}"
            lines.append(f"| {label} | {dataset} | {stat['examples']} | {100*stat['wer']:.3f} | {recall} |")
    (args.output_dir / "EXTERNAL_METRICS.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
