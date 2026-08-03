#!/usr/bin/env python3
"""Plot two-draw crossed Qwen FLEURS transfer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.report.read_text())
    keys = ("fleurs-en-us", "fleurs-de-de", "fleurs-fr-fr", "fleurs-es-419", "fleurs-pt-br")
    labels = ("English", "German", "French", "Spanish", "Portuguese")
    uniform = np.asarray([100 * payload["observed"][key]["uniform_wer_draw_mean"] for key in keys])
    task = np.asarray([100 * payload["observed"][key]["task_density_wer_draw_mean"] for key in keys])
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(8.25, 3.35),
                                   gridspec_kw={"width_ratios": [1.1, 1]})
    x, width = np.arange(len(keys)), .34
    ax0.bar(x - width / 2, uniform, width, color="#4C78A8", label="Uniform GPTQ")
    ax0.bar(x + width / 2, task, width, color="#E45756", label="Bounded task density")
    ax0.set_xticks(x, labels, rotation=20, ha="right")
    ax0.set_ylabel("Draw-mean Unicode WER (%)")
    ax0.set_title("(a) Five-language external transfer", loc="left", fontweight="bold")
    ax0.grid(axis="y", color="#dddddd", linewidth=.7)
    ax0.legend(frameon=False, fontsize=8)
    order = (*keys, "macro")
    stats = np.asarray([[100 * (payload["macro"] if key == "macro" else payload["bootstrap"][key])[field]
                         for field in ("estimate", "ci95_lower", "ci95_upper")] for key in order])
    y = np.arange(len(order))[::-1]
    ax1.axvline(0, color="#777777", linewidth=.8)
    for yi, row in zip(y, stats):
        color = "#B22222" if row[2] < 0 else "#777777"
        ax1.errorbar(row[0], yi, xerr=[[row[0] - row[1]], [row[2] - row[0]]],
                     fmt="o", color=color, ecolor=color, capsize=3, linewidth=1.2)
    ax1.set_yticks(y, (*labels, "Macro"))
    ax1.set_xlabel("Task density - uniform WER (pp)")
    ax1.set_title("(b) Crossed draw x utterance 95% CI", loc="left", fontweight="bold")
    ax1.grid(axis="x", color="#dddddd", linewidth=.7)
    fig.suptitle(f"Qwen-0.6B W3 task-density transfer on FLEURS ({payload['draws']} draws)",
                 y=1.03, fontsize=11, fontweight="bold")
    fig.tight_layout()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        kwargs = {"dpi": 240} if suffix == "png" else {}
        fig.savefig(args.output_dir / f"qwen_fleurs_transfer.{suffix}",
                    bbox_inches="tight", **kwargs)


if __name__ == "__main__":
    main()
