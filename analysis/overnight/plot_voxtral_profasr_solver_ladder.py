#!/usr/bin/env python3
"""Plot exact-target Voxtral ProfASR W2/W3 solver-basin ladder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def macro(path: Path) -> tuple[float, float]:
    row = json.loads(path.read_text(encoding="utf-8"))["macro"]
    baseline = row.get("baseline_wer_draw_mean", row.get("uniform_wer_draw_mean"))
    candidate = row.get(
        "candidate_wer_draw_mean", row.get("task_density_wer_draw_mean")
    )
    if baseline is None or candidate is None:
        raise KeyError(f"Unrecognized crossed-report macro schema in {path}")
    return 100 * baseline, 100 * candidate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fp16-metrics", type=Path, required=True)
    for bits in (2, 3):
        for method in ("task", "rtn", "awq"):
            for metric in ("raw", "capped"):
                parser.add_argument(f"--w{bits}-{method}-{metric}", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    fp16_payload = json.loads(args.fp16_metrics.read_text(encoding="utf-8"))
    fp16 = 100 * fp16_payload["evaluations"][0]["wer"]
    methods = ("Uniform GPTQ", "Task GPTQ", "RTN", "AWQ")
    values: dict[str, dict[int, list[float]]] = {"raw": {}, "capped": {}}
    for metric in ("raw", "capped"):
        for bits in (2, 3):
            task = macro(getattr(args, f"w{bits}_task_{metric}"))
            rtn = macro(getattr(args, f"w{bits}_rtn_{metric}"))
            awq = macro(getattr(args, f"w{bits}_awq_{metric}"))
            values[metric][bits] = [task[0], task[1], rtn[1], awq[1]]

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9.5,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
    })
    fig = plt.figure(figsize=(8.6, 5.4), constrained_layout=False)
    grid = fig.add_gridspec(2, 2, height_ratios=(1.05, 1.9), hspace=0.08, wspace=0.18)
    colors = {2: "#3568a8", 3: "#e08b3e"}
    markers = {2: "o", 3: "s"}
    x = np.arange(len(methods), dtype=float)
    axes = {}

    for col, metric in enumerate(("raw", "capped")):
        top = fig.add_subplot(grid[0, col])
        bottom = fig.add_subplot(grid[1, col], sharex=top)
        axes[(metric, "top")], axes[(metric, "bottom")] = top, bottom
        for bits in (2, 3):
            y = np.asarray(values[metric][bits])
            for ax in (top, bottom):
                ax.plot(x, y, color=colors[bits], marker=markers[bits], linestyle="",
                        markersize=6, label=f"W{bits}", zorder=3)
        bottom.axhline(fp16, color="#555555", linestyle="--", linewidth=1.2,
                       label="FP16")

        bottom.set_ylim(6.2, 8.15)
        collapse = [values[metric][2][2], values[metric][2][3]]
        if metric == "raw":
            top.set_ylim(90, max(collapse) * 1.08)
        else:
            top.set_ylim(90, 102.0)
        top.set_title("Raw corpus WER" if metric == "raw" else "Capped WER",
                      loc="left", fontweight="bold")
        top.spines["bottom"].set_visible(False)
        bottom.spines["top"].set_visible(False)
        top.tick_params(labelbottom=False, bottom=False)
        bottom.set_xticks(x, methods)
        for ax in (top, bottom):
            ax.grid(axis="y", alpha=0.22, linewidth=0.7)
            ax.set_axisbelow(True)
            ax.spines["right"].set_visible(False)
        diag = 0.012
        kwargs = dict(color="#333333", clip_on=False, linewidth=1.0)
        top.plot((-diag, +diag), (-diag, +diag), transform=top.transAxes, **kwargs)
        top.plot((1-diag, 1+diag), (-diag, +diag), transform=top.transAxes, **kwargs)
        bottom.plot((-diag, +diag), (1-diag, 1+diag), transform=bottom.transAxes, **kwargs)
        bottom.plot((1-diag, 1+diag), (1-diag, 1+diag), transform=bottom.transAxes, **kwargs)

        for xpos, value in enumerate(values[metric][2]):
            ax = bottom if value < 20 else top
            offset = 0.08 if ax is bottom else 0.02 * (top.get_ylim()[1] - top.get_ylim()[0])
            ax.text(xpos, value + offset, f"{value:.1f}", ha="center", va="bottom",
                    fontsize=7.5, color=colors[2])
        for xpos, value in enumerate(values[metric][3]):
            bottom.text(xpos, value - 0.10, f"{value:.1f}", ha="center", va="top",
                        fontsize=7.5, color=colors[3])

    axes[("raw", "bottom")].set_ylabel("ProfASR-v2 WER (%)")
    axes[("raw", "top")].set_ylabel("Collapse range")
    handles = [
        plt.Line2D([], [], color=colors[bits], marker=markers[bits], linestyle="",
                   label=f"W{bits}") for bits in (2, 3)
    ] + [plt.Line2D([], [], color="#555555", linestyle="--", label="FP16")]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 1.01))
    fig.suptitle("Voxtral encoder solver basin on collision-free ProfASR-v2",
                 y=1.055, fontsize=12, fontweight="bold")
    fig.subplots_adjust(top=0.86, bottom=0.12, left=0.09, right=0.98)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(args.output_dir / f"voxtral_profasr_solver_ladder.{suffix}",
                    dpi=220, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
