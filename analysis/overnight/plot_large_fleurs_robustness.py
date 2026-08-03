#!/usr/bin/env python3
"""Plot raw and per-utterance-capped Large-v3 FLEURS effects."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--capped", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    raw = json.loads(args.raw.read_text())
    capped = json.loads(args.capped.read_text())
    if raw["draws"] != capped["draws"]:
        raise ValueError("Raw and capped reports use different draw counts")
    keys = ("fleurs-en-us", "fleurs-de-de", "fleurs-fr-fr",
            "fleurs-es-419", "fleurs-pt-br", "macro")
    labels = ("English", "German", "French", "Spanish", "Portuguese", "Macro")

    def stats(payload, key):
        row = payload["macro"] if key == "macro" else payload["bootstrap"][key]
        return np.asarray([row[field] * 100 for field in
                           ("estimate", "ci95_lower", "ci95_upper")])

    raw_stats = np.stack([stats(raw, key) for key in keys])
    cap_stats = np.stack([stats(capped, key) for key in keys])
    raw_draw = np.asarray(raw["macro_delta_by_draw"]) * 100
    cap_draw = np.asarray(capped["macro_delta_by_draw"]) * 100

    plt.rcParams.update({"font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False})
    fig, (ax0, ax1) = plt.subplots(
        1, 2, figsize=(8.25, 3.45), gridspec_kw={"width_ratios": [1.4, 1]}
    )
    y = np.arange(len(keys))[::-1]
    offset = .12
    ax0.axvline(0, color="#777777", linewidth=.8)
    for rows, shift, color, label in (
        (raw_stats, +offset, "#E45756", "Corpus WER"),
        (cap_stats, -offset, "#4C78A8", "Capped WER"),
    ):
        ax0.errorbar(
            rows[:, 0], y + shift,
            xerr=[rows[:, 0] - rows[:, 1], rows[:, 2] - rows[:, 0]],
            fmt="o", color=color, ecolor=color, capsize=3, linewidth=1.15,
            label=label,
        )
    ax0.set_yticks(y, labels)
    ax0.set_xlabel("Task density - uniform WER (pp)")
    ax0.set_title("(a) Language effects and tail robustness", loc="left", fontweight="bold")
    ax0.grid(axis="x", color="#dddddd", linewidth=.7)
    ax0.legend(frameon=False, fontsize=8)

    x = np.arange(len(raw_draw))
    ax1.axhline(0, color="#777777", linewidth=.8)
    ax1.plot(x, raw_draw, marker="o", color="#E45756", linewidth=1.7,
             label="Corpus WER")
    ax1.plot(x, cap_draw, marker="o", color="#4C78A8", linewidth=1.7,
             label="Capped WER")
    for index, (a, b) in enumerate(zip(raw_draw, cap_draw, strict=True)):
        ax1.plot([index, index], [a, b], color="#bbbbbb", linewidth=.8, zorder=0)
    ax1.set_xticks(x, [f"Draw {i + 1}" for i in x])
    ax1.set_ylabel("Equal-language macro effect (pp)")
    ax1.set_title("(b) Calibration-draw stability", loc="left", fontweight="bold")
    ax1.grid(axis="y", color="#dddddd", linewidth=.7)
    ax1.legend(frameon=False, fontsize=8)

    fig.suptitle(
        f"Whisper Large-v3 W2 on FLEURS ({raw['draws']} calibration draws)",
        y=1.03, fontsize=11, fontweight="bold",
    )
    fig.tight_layout()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        kwargs = {"dpi": 240} if suffix == "png" else {}
        fig.savefig(args.output_dir / f"large_fleurs_robustness.{suffix}",
                    bbox_inches="tight", **kwargs)


if __name__ == "__main__":
    main()
