#!/usr/bin/env python3
"""Plot terminal Qwen-0.6B FLEURS method context without hiding collapse."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


LABELS = {
    "fp16": "FP16",
    "uniform": "Uniform\nGPTQ W3",
    "task": "Task-measure\nGPTQ W3",
    "rtn": "RTN W3",
    "prompt_awq": "AWQ W3\nprompt map",
    "deploy_awq": "AWQ W3\ndeployment map",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    payload = json.loads(args.summary.read_text(encoding="utf-8"))
    rows = {row["label"]: row for row in payload["methods"]}
    panels = [
        (["fp16", "uniform", "task"], (0, 27), "Decodable regime"),
        (["rtn", "prompt_awq", "deploy_awq"], (90, 122), "Collapsed neighboring controls"),
    ]

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
    })
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.55), constrained_layout=True)
    colors = {"raw": "#3568a8", "capped": "#e08b3e"}
    width = 0.34
    for ax, (keys, ylim, title) in zip(axes, panels):
        x = np.arange(len(keys), dtype=float)
        raw = np.asarray([100 * rows[k]["raw_macro_draw_mean"] for k in keys])
        capped = np.asarray([100 * rows[k]["capped_macro_draw_mean"] for k in keys])
        bars_raw = ax.bar(x - width / 2, raw, width, color=colors["raw"], label="Raw WER")
        bars_cap = ax.bar(x + width / 2, capped, width, color=colors["capped"], label="Capped WER")
        ax.set_xticks(x, [LABELS[k] for k in keys])
        ax.set_ylim(*ylim)
        ax.set_title(title, loc="left", fontweight="bold")
        ax.grid(axis="y", alpha=0.25, linewidth=0.7)
        ax.set_axisbelow(True)
        for bars in (bars_raw, bars_cap):
            for bar in bars:
                y = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    y + 0.6,
                    f"{y:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    rotation=0,
                )
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    axes[0].set_ylabel("Equal-language macro WER (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 1.07))
    fig.suptitle("Qwen-0.6B on official five-language FLEURS", y=1.13,
                 fontsize=12, fontweight="bold")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(args.output_dir / f"qwen_fleurs_method_context.{suffix}",
                    dpi=220, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
