#!/usr/bin/env python3
"""Paper figure for the Qwen prompt-template x support x density factorial."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main():
    args = arguments()
    report = json.loads(args.report.read_text())
    datasets = ("librispeech-other", "voxpopuli", "gigaspeech")
    combinations = ("00", "01", "10", "11")
    labels = ("Minimal\nprompt rows", "Minimal\nfull rows",
              "Inference\nprompt rows", "Inference\nfull rows")
    macro = {
        bits: 100 * np.mean([report["cell_wer"][dataset][bits] for dataset in datasets])
        for bits in (f"{pair}{density}" for pair in combinations for density in (0, 1))
    }

    plt.rcParams.update({"font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False})
    fig, (ax0, ax1) = plt.subplots(
        1, 2, figsize=(8.5, 3.45), gridspec_kw={"width_ratios": [1.05, 1.15]})

    x = np.asarray([0, 1.25, 3.0, 4.25])
    width = .15
    for density, color, marker, label in (
        (0, "#4C78A8", "o", "Uniform GPTQ"),
        (1, "#E45756", "D", "Bounded task density"),
    ):
        values = [macro[f"{pair}{density}"] for pair in combinations]
        ax0.scatter(x + (density * 2 - 1) * width, values, s=35, marker=marker,
                    color=color, label=label, zorder=3)
        for center, value in zip(x, values):
            ax0.plot([center, center + (density * 2 - 1) * width], [value, value],
                     color=color, linewidth=.9, alpha=.8)
    for center, pair in zip(x, combinations):
        values = [macro[f"{pair}{density}"] for density in (0, 1)]
        ax0.plot([center - width, center + width], values, color="#999999",
                 linewidth=.8, zorder=1)
    ax0.axvline(2.125, color="#bbbbbb", linewidth=.9, linestyle="--")
    ax0.set_xticks(x, labels)
    ax0.set_ylabel("Hard-domain macro WER (%)")
    ax0.set_title("(a) Eight matched intervention cells", loc="left", fontweight="bold")
    ax0.grid(axis="y", color="#dddddd", linewidth=.7)
    ax0.legend(frameon=False, fontsize=8)

    contrast_order = (
        "template", "support", "density", "template:support",
        "template:density", "support:density", "template:support:density")
    contrast_labels = (
        "Template", "Row support", "Task density", "Template x support",
        "Template x density", "Support x density", "Three-way")
    stats = np.asarray([
        [100 * report["macro_contrasts"][key][field]
         for field in ("estimate", "ci95_lower", "ci95_upper")]
        for key in contrast_order
    ])
    y = np.arange(len(contrast_order))[::-1]
    ax1.axvline(0, color="#777777", linewidth=.8)
    colors = ["#4C78A8"] * 3 + ["#7F7F7F"] * 4
    for yi, row, color in zip(y, stats, colors):
        ax1.errorbar(row[0], yi, xerr=[[row[0] - row[1]], [row[2] - row[0]]],
                     fmt="o", color=color, ecolor=color, capsize=3, linewidth=1.2)
    ax1.set_yticks(y, contrast_labels)
    ax1.set_xlabel("Factorial effect on WER (pp; lower is better)")
    ax1.set_title("(b) Paired utterance-bootstrap 95% CI", loc="left", fontweight="bold")
    ax1.grid(axis="x", color="#dddddd", linewidth=.7)

    fig.suptitle("Deployment-conditioned calibration states", y=1.03,
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        kwargs = {"dpi": 240} if suffix == "png" else {}
        fig.savefig(args.output_dir / f"interface_factorial.{suffix}",
                    bbox_inches="tight", **kwargs)
    print(args.output_dir / "interface_factorial.pdf")


if __name__ == "__main__":
    main()
