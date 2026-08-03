#!/usr/bin/env python3
"""Plot Voxtral bit-width x target x density interaction forests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROWS = (
    ("W2 · FLEURS density", "density", "W2", "FLEURS"),
    ("W2 · ProfASR density", "density", "W2", "ProfASR"),
    ("W2 · target interaction", "target", "W2", None),
    ("W3 · FLEURS density", "density", "W3", "FLEURS"),
    ("W3 · ProfASR density", "density", "W3", "ProfASR"),
    ("W3 · target interaction", "target", "W3", None),
    ("Bit × target × density", "threeway", None, None),
)


def load_rows(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    out = []
    for label, kind, bit, target in ROWS:
        if kind == "density":
            stat = payload["density_effects"][bit][target]
        elif kind == "target":
            stat = payload["target_interactions"][bit]
        else:
            stat = payload["bit_target_interaction"]
        out.append({"label": label, "kind": kind, "bit": bit, **stat})
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--capped", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    panels = (("Raw corpus WER", load_rows(args.raw)),
              ("Capped WER", load_rows(args.capped)))

    colors = {"W2": "#3568a8", "W3": "#e08b3e", None: "#222222"}
    markers = {"density": "o", "target": "D", "threeway": "*"}
    y = np.arange(len(ROWS))[::-1]
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9.5,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
    })
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.5), sharey=True,
                             constrained_layout=True)
    for ax, (title, rows) in zip(axes, panels):
        estimates = 100 * np.asarray([row["estimate"] for row in rows])
        lower = 100 * np.asarray([row["ci95_lower"] for row in rows])
        upper = 100 * np.asarray([row["ci95_upper"] for row in rows])
        for ypos, row, estimate, lo, hi in zip(y, rows, estimates, lower, upper):
            color = colors[row["bit"]]
            ax.errorbar(
                estimate, ypos,
                xerr=np.asarray([[estimate - lo], [hi - estimate]]),
                fmt=markers[row["kind"]], color=color, markerfacecolor=color,
                markeredgecolor="white", markeredgewidth=0.6,
                markersize=8 if row["kind"] == "threeway" else 6,
                capsize=2.5, linewidth=1.5, zorder=3,
            )
            span = max(upper.max() - lower.min(), 1.0)
            ax.text(hi + .025 * span, ypos, f"{estimate:+.2f}", ha="left",
                    va="center", fontsize=7.5, color=color)
        ax.axvline(0, color="#555555", linestyle="--", linewidth=1.0)
        ax.axhline(3.5, color="#aaaaaa", linewidth=0.7)
        ax.axhline(0.5, color="#aaaaaa", linewidth=0.7)
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_xlabel("WER contrast (percentage points)")
        ax.grid(axis="x", alpha=0.2, linewidth=0.7)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_yticks(y, [row[0] for row in ROWS])
    axes[0].text(
        0.01, -0.18,
        "Density: task − uniform.  Target: ProfASR − FLEURS.  Three-way: W3 − W2 target interaction.",
        transform=axes[0].transAxes, ha="left", va="top", fontsize=8,
    )
    fig.suptitle("Voxtral target dependence disappears at W3 saturation",
                 fontsize=12, fontweight="bold")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(args.output_dir / f"voxtral_bit_target_interaction.{suffix}",
                    dpi=220, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
