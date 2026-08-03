#!/usr/bin/env python3
"""Forest plot of frozen crossed-draw bounded-KL efficacy endpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


SERIES = (
    ("Whisper Small W2", "small_klcap_crossed_3draw"),
    ("Distil-Large-v3 W2", "distil_klcap_crossed_3draw"),
    ("Whisper Large-v3 W2", "large_klcap_crossed_3draw"),
    ("Qwen-0.6B text W3", "qwen06_klcap_plain_crossed_6draw"),
    ("Voxtral Mini W2", "voxtral_klcap_plain_crossed_11draw"),
    ("Whisper Medium W2", "medium_klcap_crossed_batch8_3draw"),
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for label, subdir in SERIES:
        payload = json.loads((args.analysis_root / subdir / "crossed_multidraw_bootstrap.json").read_text())
        stat = payload["macro"]
        rows.append((label, 100 * stat["estimate"], 100 * stat["ci95_lower"],
                     100 * stat["ci95_upper"], payload["draws"]))

    plt.rcParams.update({"font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False})
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(8.6, 3.45),
                                   gridspec_kw={"width_ratios": [.8, 1.35]})
    all_stats = np.asarray([[row[1], row[2], row[3]] for row in rows])

    # The Small stress endpoint needs a separate scale; the second panel shows
    # the deployment-usable models and Medium null without hiding their CIs.
    row = all_stats[0]
    ax0.axvline(0, color="#777777", linewidth=.8)
    ax0.errorbar(row[0], 0, xerr=[[row[0] - row[1]], [row[2] - row[0]]],
                 fmt="o", color="#B22222", ecolor="#B22222", capsize=3,
                 linewidth=1.4)
    ax0.set_yticks([0], [f"{rows[0][0]}\n({rows[0][4]} draws)"])
    ax0.set_xlim(-18, 1)
    ax0.set_xlabel("KL - uniform WER (pp)")
    ax0.set_title("(a) Stress regime", loc="left", fontweight="bold")
    ax0.grid(axis="x", color="#dddddd", linewidth=.7)

    stats = all_stats[1:]
    y = np.arange(len(stats))[::-1]
    ax1.axvline(0, color="#777777", linewidth=.8)
    colors = ["#B22222" if upper < 0 else "#777777" for _, _, upper in stats]
    for yi, row, color in zip(y, stats, colors):
        ax1.errorbar(row[0], yi, xerr=[[row[0] - row[1]], [row[2] - row[0]]],
                     fmt="o", color=color, ecolor=color, capsize=3, linewidth=1.3)
    ax1.set_yticks(y, [f"{label}\n({draws} draws)" for label, *_, draws in rows[1:]])
    ax1.set_xlim(-2.15, .75)
    ax1.set_xlabel("Bounded task density - uniform GPTQ WER (pp)")
    ax1.set_title("(b) Recoverable and null regimes", loc="left", fontweight="bold")
    ax1.grid(axis="x", color="#dddddd", linewidth=.7)

    fig.suptitle("Cross-family low-bit efficacy with crossed draw x utterance inference",
                 y=1.03, fontsize=11, fontweight="bold")
    fig.tight_layout()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        kwargs = {"dpi": 240} if suffix == "png" else {}
        fig.savefig(args.output_dir / f"cross_family_efficacy.{suffix}",
                    bbox_inches="tight", **kwargs)
    print(args.output_dir / "cross_family_efficacy.pdf")


if __name__ == "__main__":
    main()
