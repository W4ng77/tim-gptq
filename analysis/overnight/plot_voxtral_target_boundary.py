#!/usr/bin/env python3
"""Plot Voxtral's same-model FLEURS/ProfASR target-distribution boundary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def values(row: dict) -> np.ndarray:
    return 100 * np.asarray([row["estimate"], row["ci95_lower"], row["ci95_upper"]])


def point(ax, row: np.ndarray, y: float, color: str, marker: str, label: str | None):
    ax.errorbar(
        row[0], y, xerr=[[row[0] - row[1]], [row[2] - row[0]]],
        fmt=marker, color=color, ecolor=color, capsize=3,
        linewidth=1.3, markersize=6, label=label,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--capped", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    raw, capped = load(args.raw), load(args.capped)

    plt.rcParams.update({
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(8.1, 3.2))
    for ax in (ax0, ax1):
        ax.axvline(0, color="#777777", linewidth=.9)
        ax.grid(axis="x", color="#dddddd", linewidth=.7)

    labels = ("FLEURS", "ProfASR-v2")
    y = np.asarray([1.0, 0.0])
    for i, target in enumerate(labels):
        point(ax0, values(raw["target_effects"][target]), y[i] + .10,
              "#B22222", "o", "Raw WER" if i == 0 else None)
        point(ax0, values(capped["target_effects"][target]), y[i] - .10,
              "#315A8A", "s", "Capped WER" if i == 0 else None)
    ax0.set_yticks(y, labels)
    ax0.set_xlabel("Task density − uniform WER (pp; lower is better)")
    ax0.set_title("(a) Same estimator, opposite targets", loc="left", fontweight="bold")
    ax0.legend(frameon=False, fontsize=8, loc="upper right")

    point(ax1, values(raw["interaction"]), .60, "#B22222", "o", None)
    point(ax1, values(capped["interaction"]), .20, "#315A8A", "s", None)
    ax1.set_yticks([.60, .20], ["Raw", "Capped"])
    ax1.set_xlabel("ProfASR − FLEURS interaction (pp)")
    ax1.set_title("(b) Shared-draw target interaction", loc="left", fontweight="bold")
    ax1.set_ylim(-.05, .85)

    fig.suptitle(
        "Voxtral W2: task-measure efficacy belongs to the model–target pair",
        y=1.04, fontsize=11, fontweight="bold",
    )
    fig.tight_layout()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        kwargs = {"dpi": 240} if suffix == "png" else {}
        fig.savefig(
            args.output_dir / f"voxtral_target_boundary.{suffix}",
            bbox_inches="tight", **kwargs,
        )


if __name__ == "__main__":
    main()
