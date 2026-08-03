#!/usr/bin/env python3
"""Plot crossed ContextASR WER and entity recall across three model maps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


SERIES = (
    ("Qwen-0.6B", "qwen06_contextasr_crossed_3draw",
     "crossed_external_bootstrap.json", "crossed_entity_bootstrap.json"),
    ("Qwen-1.7B", "qwen17_contextasr_crossed_3draw",
     "crossed_external_bootstrap.json", "crossed_entity_bootstrap.json"),
    ("Whisper Large-v3", "large_contextasr_crossed_3draw",
     "crossed_external_bootstrap.json", "crossed_entity_bootstrap.json"),
    ("Voxtral Mini", "voxtral_contextasr_crossed_3draw",
     "crossed_external_contrast.json", "../voxtral_contextasr_entity_crossed_3draw/crossed_entity_contrast.json"),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    for label, subdir, wer_name, entity_name in SERIES:
        wer = json.loads((args.analysis_root / subdir / wer_name).read_text())["macro"]
        entity = json.loads((args.analysis_root / subdir / entity_name).read_text())
        rows.append((label, wer, entity))

    plt.rcParams.update({
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(7.8, 3.25), sharey=True)
    y = np.arange(len(rows))[::-1]
    for ax, index, title, xlabel in (
        (ax0, 1, "(a) Unicode WER", "Task density - uniform WER (pp)"),
        (ax1, 2, "(b) Exact entity recall", "Task density - uniform recall (pp)"),
    ):
        ax.axvline(0, color="#777777", linewidth=.8)
        for yi, row in zip(y, rows):
            stat = row[index]
            estimate, lower, upper = (100 * stat[key] for key in
                                      ("estimate", "ci95_lower", "ci95_upper"))
            passes = upper < 0 if index == 1 else lower > 0
            color = "#B22222" if passes else "#777777"
            ax.errorbar(estimate, yi,
                        xerr=[[estimate - lower], [upper - estimate]],
                        fmt="o", color=color, ecolor=color, capsize=3,
                        linewidth=1.3, markersize=6)
        ax.set_yticks(y, [row[0] for row in rows])
        ax.set_xlabel(xlabel)
        ax.set_title(title, loc="left", fontweight="bold")
        ax.grid(axis="x", color="#dddddd", linewidth=.7)

    fig.suptitle("ContextASR task-measure transfer depends on the model-induced map",
                 y=1.03, fontsize=11, fontweight="bold")
    fig.tight_layout()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        kwargs = {"dpi": 240} if suffix == "png" else {}
        fig.savefig(args.output_dir / f"contextasr_cross_model.{suffix}",
                    bbox_inches="tight", **kwargs)


if __name__ == "__main__":
    main()
