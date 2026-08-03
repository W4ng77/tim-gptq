#!/usr/bin/env python3
"""Plot exact-target Large-v3 FLEURS W2/W3/W4 task-measure ladder."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from summarize_fleurs_methods import run_macro


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fp16", type=Path, required=True)
    parser.add_argument(
        "--pair", nargs=3, action="append", required=True,
        metavar=("BITS", "UNIFORM_RUN", "TASK_RUN"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    grouped: dict[int, list[tuple[Path, Path]]] = defaultdict(list)
    for bits, uniform, task in args.pair:
        grouped[int(bits)].append((Path(uniform), Path(task)))
    bit_order = sorted(grouped, reverse=True)
    fp16 = {metric: 100 * run_macro(args.fp16, metric == "capped")
            for metric in ("raw", "capped")}

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
    })
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.55), sharey=True,
                             constrained_layout=True)
    colors = {"uniform": "#3568a8", "task": "#e08b3e"}
    markers = {"uniform": "s", "task": "o"}
    x = np.arange(len(bit_order), dtype=float)

    for ax, metric in zip(axes, ("raw", "capped")):
        capped = metric == "capped"
        for method_index, method in enumerate(("uniform", "task")):
            means, draws = [], []
            for bits in bit_order:
                values = [
                    100 * run_macro(pair[0 if method == "uniform" else 1], capped)
                    for pair in grouped[bits]
                ]
                draws.append(values)
                means.append(float(np.mean(values)))
            offset = -0.08 if method == "uniform" else 0.08
            ax.plot(
                x + offset, means, color=colors[method], marker=markers[method],
                linewidth=2, markersize=6,
                label="Uniform GPTQ" if method == "uniform" else "Task-measure GPTQ",
            )
            for xpos, values in zip(x + offset, draws):
                jitter = np.linspace(-0.025, 0.025, len(values)) if len(values) > 1 else [0]
                ax.scatter(
                    xpos + jitter, values, s=20, color=colors[method], alpha=0.55,
                    edgecolor="white", linewidth=0.4, zorder=3,
                )
        ax.axhline(fp16[metric], color="#555555", linestyle="--", linewidth=1.4,
                   label="FP16 reference")
        for xpos, bits in zip(x, bit_order):
            u = np.mean([100 * run_macro(pair[0], capped) for pair in grouped[bits]])
            t = np.mean([100 * run_macro(pair[1], capped) for pair in grouped[bits]])
            ax.text(xpos, max(u, t) + 0.7, f"Δ {t-u:+.2f}", ha="center",
                    va="bottom", fontsize=8)
        ax.set_xticks(x, [f"W{bits}" for bits in bit_order])
        ax.set_title("Raw corpus WER" if metric == "raw" else "Capped WER", loc="left",
                     fontweight="bold")
        ax.grid(axis="y", alpha=0.25, linewidth=0.7)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    axes[0].set_ylabel("Equal-language macro WER (%)")
    axes[0].set_ylim(bottom=0)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 1.05))
    fig.suptitle("Whisper Large-v3 encoder on official five-language FLEURS",
                 y=1.12, fontsize=12, fontweight="bold")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(args.output_dir / f"large_fleurs_bitwidth_ladder.{suffix}",
                    dpi=220, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
