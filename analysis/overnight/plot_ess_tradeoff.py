#!/usr/bin/env python3
"""Plot bounded density versus ESS=.8 across models and deployment targets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def effect(path: Path) -> tuple[float, float, float]:
    row = json.loads(path.read_text(encoding="utf-8"))["macro"]
    return tuple(100 * row[key] for key in ("estimate", "ci95_lower", "ci95_upper"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for metric in ("raw", "capped"):
        for model in ("vox", "large"):
            for target in ("prof", "fleurs"):
                for method in ("bounded", "ess"):
                    parser.add_argument(
                        f"--{metric}-{model}-{target}-{method}",
                        type=Path,
                        required=True,
                    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9.5,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
    })
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 5.5), sharey=True)
    colors = {"bounded": "#3568a8", "ess": "#e08b3e"}
    labels = {"bounded": "Bounded density", "ess": "ESS≥.8 contraction"}
    rows = (
        ("vox", "prof", "Voxtral · ProfASR-v2"),
        ("vox", "fleurs", "Voxtral · FLEURS"),
        ("large", "prof", "Large-v3 · ProfASR-v2"),
        ("large", "fleurs", "Large-v3 · FLEURS"),
    )
    offsets = {"bounded": -0.11, "ess": 0.11}

    for ax, metric in zip(axes, ("raw", "capped"), strict=True):
        for row_index, (model_key, target_key, _) in enumerate(rows):
            for method in ("bounded", "ess"):
                estimate, lower, upper = effect(
                    getattr(args, f"{metric}_{model_key}_{target_key}_{method}")
                )
                y = row_index + offsets[method]
                ax.errorbar(
                    estimate,
                    y,
                    xerr=[[estimate - lower], [upper - estimate]],
                    fmt="o",
                    color=colors[method],
                    capsize=3,
                    markersize=5.5,
                    linewidth=1.5,
                    label=labels[method] if row_index == 0 else None,
                    zorder=3,
                )
                # Put the two labels on opposite sides of each row.  Point
                # offsets remain legible when estimates cluster around zero.
                label_offset = (0, -15) if method == "bounded" else (0, 15)
                ax.annotate(
                    f"{estimate:+.2f}",
                    xy=(estimate, y),
                    xytext=label_offset,
                    textcoords="offset points",
                    ha="center",
                    va="center",
                    fontsize=7.5,
                    color=colors[method],
                )
        ax.axvline(0, color="#555555", linestyle="--", linewidth=1)
        ax.axhline(1.5, color="#999999", linewidth=0.7, alpha=0.55)
        ax.grid(axis="x", alpha=0.22, linewidth=0.7)
        ax.set_title("Raw corpus WER" if metric == "raw" else "Capped WER", loc="left")
        ax.set_xlabel("Candidate − uniform GPTQ (WER pp)\n← improves    harms →")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_yticks(range(len(rows)), [label for _, _, label in rows])
    axes[0].invert_yaxis()
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.925),
        ncol=2,
        frameon=False,
    )
    fig.suptitle(
        "ESS contraction is model–target conditioned, not a monotone safety guarantee",
        y=0.99,
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.83))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for stem in ("ess_tradeoff_crossarch", "voxtral_ess_tradeoff"):
        for suffix in ("png", "pdf"):
            fig.savefig(
                args.output_dir / f"{stem}.{suffix}",
                dpi=220,
                bbox_inches="tight",
            )
    plt.close(fig)


if __name__ == "__main__":
    main()
