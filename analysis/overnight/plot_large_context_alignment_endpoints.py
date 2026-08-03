#!/usr/bin/env python3
"""Plot the terminal Large-v3 ContextASR alignment intervention on both endpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def triple(path: Path, nested: str | None = None) -> np.ndarray:
    row = json.loads(path.read_text())
    if nested is not None:
        row = row[nested]
    return 100 * np.asarray([row["estimate"], row["ci95_lower"], row["ci95_upper"]])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.analysis_root

    # Convert WER to accuracy orientation so positive always means better.
    aligned_wer = -triple(
        root / "large_contextasr_crossed_3draw/crossed_external_bootstrap.json", "macro"
    )[[0, 2, 1]]
    perm_aligned_wer = -triple(
        root / "large_context_permuted_vs_aligned_3draw/crossed_external_contrast.json", "macro"
    )[[0, 2, 1]]
    perm_uniform_wer = -triple(
        root / "large_context_permuted_vs_uniform_3draw/crossed_external_contrast.json", "macro"
    )[[0, 2, 1]]
    aligned_entity = triple(
        root / "large_contextasr_crossed_3draw/crossed_entity_bootstrap.json"
    )
    perm_aligned_entity = triple(
        root / "large_context_permuted_entity_vs_aligned_3draw/crossed_entity_contrast.json"
    )
    perm_uniform_entity = triple(
        root / "large_context_permuted_entity_vs_uniform_3draw/crossed_entity_contrast.json"
    )

    labels = ("Aligned − uniform", "Permuted − aligned", "Permuted − uniform")
    wer = np.stack((aligned_wer, perm_aligned_wer, perm_uniform_wer))
    entity = np.stack((aligned_entity, perm_aligned_entity, perm_uniform_entity))

    plt.rcParams.update({
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.25), sharey=True)
    y = np.arange(len(labels))[::-1]
    for ax, rows, title, xlabel in (
        (axes[0], wer, "(a) Recognition accuracy", "WER benefit (pp; higher is better)"),
        (axes[1], entity, "(b) Exact entity recall", "Recall benefit (pp; higher is better)"),
    ):
        ax.axvline(0, color="#777777", linewidth=.9)
        for yi, row in zip(y, rows):
            lo, hi = row[1], row[2]
            identified = lo > 0 or hi < 0
            color = "#B22222" if identified else "#777777"
            ax.errorbar(
                row[0], yi, xerr=[[row[0] - lo], [hi - row[0]]], fmt="o",
                color=color, ecolor=color, capsize=3, linewidth=1.3, markersize=6,
            )
        ax.set_yticks(y, labels)
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_xlabel(xlabel)
        ax.grid(axis="x", color="#dddddd", linewidth=.7)
    fig.suptitle(
        "Large-v3 ContextASR: alignment creates the gain; permutation removes it",
        y=1.03, fontsize=11, fontweight="bold",
    )
    fig.tight_layout()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        kwargs = {"dpi": 240} if suffix == "png" else {}
        fig.savefig(
            args.output_dir / f"large_context_alignment_endpoints.{suffix}",
            bbox_inches="tight", **kwargs,
        )


if __name__ == "__main__":
    main()
