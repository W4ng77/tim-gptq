#!/usr/bin/env python3
"""Plot the crossed Qwen model-scale by target-distribution interaction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


SOURCES = (
    ("Qwen-0.6B", "FLEURS", "qwen06_fleurs5_crossed_3draw/crossed_external_bootstrap.json"),
    ("Qwen-0.6B", "ProfASR-v2", "qwen06_profasr_v2_crossed_3draw/crossed_external_bootstrap.json"),
    ("Qwen-1.7B", "FLEURS", "qwen17_fleurs5_crossed_3draw/crossed_external_bootstrap.json"),
    ("Qwen-1.7B", "ProfASR-v2", "qwen17_profasr_v2_crossed_3draw/crossed_external_bootstrap.json"),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    values = {}
    for model, target, relative in SOURCES:
        macro = json.loads((args.analysis_root / relative).read_text())["macro"]
        values[(model, target)] = np.asarray([
            100 * macro["estimate"],
            100 * macro["ci95_lower"],
            100 * macro["ci95_upper"],
        ])
    formal = json.loads((
        args.analysis_root / "qwen_scale_target_interaction_3draw" /
        "crossed_scale_target_bootstrap.json"
    ).read_text())["interaction"]

    plt.rcParams.update({
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    fig, (ax0, ax1) = plt.subplots(
        1, 2, figsize=(8.25, 3.35), gridspec_kw={"width_ratios": [1, 1.25]}
    )
    models = ("Qwen-0.6B", "Qwen-1.7B")
    colors = {"FLEURS": "#4C78A8", "ProfASR-v2": "#E45756"}
    x = np.arange(2)

    ax0.axhline(0, color="#777777", linewidth=.8)
    for target in ("FLEURS", "ProfASR-v2"):
        stats = np.asarray([values[(model, target)] for model in models])
        ax0.plot(x, stats[:, 0], marker="o", linewidth=1.8,
                 color=colors[target], label=target)
        ax0.errorbar(x, stats[:, 0],
                     yerr=[stats[:, 0] - stats[:, 1], stats[:, 2] - stats[:, 0]],
                     fmt="none", ecolor=colors[target], capsize=3, linewidth=1.1)
    ax0.set_xticks(x, models)
    ax0.set_ylabel("Task density - uniform WER (pp)")
    ax0.set_title("(a) Crossed target response", loc="left", fontweight="bold")
    ax0.grid(axis="y", color="#dddddd", linewidth=.7)
    ax0.legend(frameon=False, fontsize=8, loc="center right")
    ax0.text(.04, .04,
             f"formal interaction = {100*formal['estimate']:+.3f} pp\n"
             f"95% CI [{100*formal['ci95_lower']:+.3f}, "
             f"{100*formal['ci95_upper']:+.3f}]",
             transform=ax0.transAxes, fontsize=8, va="bottom",
             bbox={"boxstyle": "round,pad=.3", "facecolor": "white",
                   "edgecolor": "#bbbbbb", "alpha": .9})

    ordered = (
        ("Qwen-0.6B / FLEURS", values[("Qwen-0.6B", "FLEURS")]),
        ("Qwen-0.6B / ProfASR-v2", values[("Qwen-0.6B", "ProfASR-v2")]),
        ("Qwen-1.7B / FLEURS", values[("Qwen-1.7B", "FLEURS")]),
        ("Qwen-1.7B / ProfASR-v2", values[("Qwen-1.7B", "ProfASR-v2")]),
    )
    y = np.arange(len(ordered))[::-1]
    ax1.axvline(0, color="#777777", linewidth=.8)
    for yi, (label, row) in zip(y, ordered):
        target = "FLEURS" if "FLEURS" in label else "ProfASR-v2"
        color = colors[target] if row[2] < 0 else "#777777"
        ax1.errorbar(row[0], yi,
                     xerr=[[row[0] - row[1]], [row[2] - row[0]]],
                     fmt="o", color=color, ecolor=color, capsize=3, linewidth=1.2)
    ax1.set_yticks(y, [label for label, _ in ordered])
    ax1.set_xlabel("Crossed draw x utterance effect (pp)")
    ax1.set_title("(b) 95% intervals", loc="left", fontweight="bold")
    ax1.grid(axis="x", color="#dddddd", linewidth=.7)

    fig.suptitle("Task-density efficacy depends jointly on model scale and target distribution",
                 y=1.03, fontsize=11, fontweight="bold")
    fig.tight_layout()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        kwargs = {"dpi": 240} if suffix == "png" else {}
        fig.savefig(args.output_dir / f"scale_target_interaction.{suffix}",
                    bbox_inches="tight", **kwargs)


if __name__ == "__main__":
    main()
