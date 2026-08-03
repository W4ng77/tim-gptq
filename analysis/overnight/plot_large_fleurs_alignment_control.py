#!/usr/bin/env python3
"""Plot terminal Large-v3 FLEURS alignment controls, raw and tail-capped."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load(path: Path) -> np.ndarray:
    row = json.loads(path.read_text())["macro"]
    return 100 * np.asarray([row["estimate"], row["ci95_lower"], row["ci95_upper"]])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.analysis_root
    aligned = np.stack([
        load(root / "large_fleurs_permuted_vs_aligned_3draw/crossed_external_contrast.json"),
        load(root / "large_fleurs_permuted_vs_aligned_capped_3draw/crossed_external_contrast.json"),
    ])
    uniform = np.stack([
        load(root / "large_fleurs_permuted_vs_uniform_3draw/crossed_external_contrast.json"),
        load(root / "large_fleurs_permuted_vs_uniform_capped_3draw/crossed_external_contrast.json"),
    ])

    plt.rcParams.update({"font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(6.8, 2.9))
    y = np.asarray([1.0, 0.0])
    ax.axvline(0, color="#777777", linewidth=.9)
    for rows, offset, color, marker, label in (
        (aligned, .10, "#B22222", "o", "Permuted − aligned"),
        (uniform, -.10, "#777777", "s", "Permuted − uniform"),
    ):
        ax.errorbar(
            rows[:, 0], y + offset,
            xerr=[rows[:, 0] - rows[:, 1], rows[:, 2] - rows[:, 0]],
            fmt=marker, color=color, ecolor=color, capsize=3,
            linewidth=1.3, markersize=6, label=label,
        )
    ax.set_yticks(y, ["Raw WER", "Capped WER"])
    ax.set_xlabel("Permutation-induced WER change (pp; higher is worse)")
    ax.set_title(
        "Large-v3 FLEURS: state misalignment removes the gain beyond rollout tails",
        loc="left", fontsize=11, fontweight="bold",
    )
    ax.grid(axis="x", color="#dddddd", linewidth=.7)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        kwargs = {"dpi": 240} if suffix == "png" else {}
        fig.savefig(
            args.output_dir / f"large_fleurs_alignment_control.{suffix}",
            bbox_inches="tight", **kwargs,
        )


if __name__ == "__main__":
    main()
