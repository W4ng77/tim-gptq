#!/usr/bin/env python3
"""Plot aligned-vs-permuted identification across Large-v3 and Voxtral."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


SERIES = (
    ("Whisper Large-v3", "large_permutation_vs_aligned", "large_permutation_vs_uniform"),
    ("Voxtral Mini", "voxtral_permutation_vs_aligned", "voxtral_permutation_vs_uniform"),
)


def macro(root: Path, subdir: str):
    return json.loads((root / subdir / "crossed_multidraw_bootstrap.json").read_text())["macro"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    plt.rcParams.update({
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(8.15, 3.25), sharey=True)
    y = np.arange(len(SERIES))[::-1]
    axes = (
        (ax0, 1, "Permuted - aligned task density", "(a) Break correspondence"),
        (ax1, 2, "Permuted - uniform GPTQ", "(b) Preserve only marginals"),
    )
    for ax, source_index, xlabel, title in axes:
        ax.axvline(0, color="#777777", linewidth=.8)
        for yi, row in zip(y, SERIES):
            stat = macro(args.analysis_root, row[source_index])
            estimate, lower, upper = (100 * stat[key] for key in
                                      ("estimate", "ci95_lower", "ci95_upper"))
            color = "#B22222" if lower > 0 else "#777777"
            ax.errorbar(estimate, yi,
                        xerr=[[estimate - lower], [upper - estimate]],
                        fmt="o", color=color, ecolor=color, capsize=3,
                        linewidth=1.3, markersize=6)
        ax.set_yticks(y, [row[0] for row in SERIES])
        ax.set_xlabel(f"{xlabel} WER (pp)")
        ax.set_title(title, loc="left", fontweight="bold")
        ax.grid(axis="x", color="#dddddd", linewidth=.7)

    fig.suptitle("Within-sample permutation identifies weight-state alignment",
                 y=1.03, fontsize=11, fontweight="bold")
    fig.tight_layout()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        kwargs = {"dpi": 240} if suffix == "png" else {}
        fig.savefig(args.output_dir / f"alignment_architectures.{suffix}",
                    bbox_inches="tight", **kwargs)


if __name__ == "__main__":
    main()
