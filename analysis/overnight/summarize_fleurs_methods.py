#!/usr/bin/env python3
"""Summarize absolute equal-language FLEURS WER across named completed runs."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from crossed_external_bootstrap import load_pair


DATASETS = (
    "fleurs-en-us",
    "fleurs-de-de",
    "fleurs-fr-fr",
    "fleurs-es-419",
    "fleurs-pt-br",
)


def run_macro(path: Path, capped: bool) -> float:
    status = json.loads((path / "status.json").read_text())
    if status.get("state") != "completed":
        raise ValueError(f"Run is not completed: {path}")
    values = []
    for dataset in DATASETS:
        lengths, errors, _ = load_pair(path, path, dataset, capped)
        values.append(errors.sum() / lengths.sum())
    return float(np.mean(values))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", nargs=2, action="append", required=True,
                        metavar=("LABEL", "RUN_DIR"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    grouped: dict[str, list[Path]] = defaultdict(list)
    order = []
    for label, path in args.run:
        if label not in grouped:
            order.append(label)
        grouped[label].append(Path(path))

    rows = []
    for label in order:
        raw = [run_macro(path, False) for path in grouped[label]]
        capped = [run_macro(path, True) for path in grouped[label]]
        rows.append({
            "label": label,
            "draws": len(raw),
            "raw_macro_draw_mean": float(np.mean(raw)),
            "capped_macro_draw_mean": float(np.mean(capped)),
            "raw_macro_by_draw": raw,
            "capped_macro_by_draw": capped,
            "run_dirs": [str(path) for path in grouped[label]],
        })

    payload = {
        "metric": "equal_language_macro_unicode_wer",
        "datasets": list(DATASETS),
        "methods": rows,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "fleurs_method_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    lines = [
        "# FLEURS absolute method summary", "",
        "Only completed runs; equal-language macro Unicode WER. Capped WER "
        "clips each utterance's edit count at reference length.", "",
        "| Method | Draws | Raw macro % | Capped macro % | Raw by draw % |",
        "|---|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['label']} | {row['draws']} | "
            f"{100*row['raw_macro_draw_mean']:.3f} | "
            f"{100*row['capped_macro_draw_mean']:.3f} | `"
            + "/".join(f"{100*x:.3f}" for x in row["raw_macro_by_draw"])
            + "` |"
        )
    (args.output_dir / "FLEURS_METHOD_SUMMARY.md").write_text(
        "\n".join(lines) + "\n"
    )


if __name__ == "__main__":
    main()
