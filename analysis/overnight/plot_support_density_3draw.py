#!/usr/bin/env python3
"""Plot the terminal three-draw Qwen support x density analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_macro(root: Path, subdir: str, filename: str = "crossed_multidraw_bootstrap.json"):
    return json.loads((root / subdir / filename).read_text())["macro"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    support_u = load_macro(args.analysis_root, "qwen_support_under_uniform_3draw")
    support_k = load_macro(args.analysis_root, "qwen_support_under_kl_3draw")
    density_p = load_macro(args.analysis_root, "qwen_density_on_prompt_3draw")
    density_f = load_macro(args.analysis_root, "qwen_density_on_full_3draw")
    complete = load_macro(args.analysis_root, "qwen_complete_measure_vs_prompt_uniform_3draw")
    interaction = load_macro(
        args.analysis_root,
        "qwen_support_density_3draw_interaction",
        "crossed_multidraw_factorial.json",
    )

    plt.rcParams.update({
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    fig, (ax0, ax1) = plt.subplots(
        1, 2, figsize=(8.35, 3.45), gridspec_kw={"width_ratios": [.9, 1.35]}
    )

    # Absolute draw-mean macro WER for the four deployment-map-aligned cells.
    cell = {
        (0, "Uniform"): 100 * support_u["baseline_wer_draw_mean"],
        (1, "Uniform"): 100 * support_u["candidate_wer_draw_mean"],
        (0, "Bounded KL"): 100 * density_p["candidate_wer_draw_mean"],
        (1, "Bounded KL"): 100 * density_f["candidate_wer_draw_mean"],
    }
    colors = {"Uniform": "#4C78A8", "Bounded KL": "#E45756"}
    for density in ("Uniform", "Bounded KL"):
        values = [cell[(x, density)] for x in (0, 1)]
        ax0.plot((0, 1), values, marker="o", linewidth=1.8,
                 color=colors[density], label=density)
        for x, value in enumerate(values):
            ax0.text(x, value + .035, f"{value:.3f}", color=colors[density],
                     ha="center", va="bottom", fontsize=8)
    ax0.set_xticks((0, 1), ("Prompt rows", "Full response"))
    ax0.set_xlim(-.25, 1.25)
    ax0.set_ylim(9.85, 10.48)
    ax0.set_ylabel("Three-draw macro WER (%)")
    ax0.set_title("(a) Fixed deployment map", loc="left", fontweight="bold")
    ax0.grid(axis="y", color="#dddddd", linewidth=.7)
    ax0.legend(frameon=False, fontsize=8)

    rows = (
        ("Complete measure\n(full KL - prompt uniform)", complete),
        ("Support under uniform", support_u),
        ("Support under bounded KL", support_k),
        ("Density on prompt support", density_p),
        ("Density on full support", density_f),
        ("Support x density interaction", interaction),
    )
    y = np.arange(len(rows))[::-1]
    ax1.axvline(0, color="#777777", linewidth=.8)
    for yi, (_, stat) in zip(y, rows):
        estimate, lower, upper = (100 * stat[key] for key in
                                  ("estimate", "ci95_lower", "ci95_upper"))
        color = "#B22222" if upper < 0 else "#777777"
        ax1.errorbar(estimate, yi,
                     xerr=[[estimate - lower], [upper - estimate]],
                     fmt="o", color=color, ecolor=color, capsize=3, linewidth=1.2)
    ax1.set_yticks(y, [label for label, _ in rows])
    ax1.set_xlabel("WER contrast (pp), crossed 95% CI")
    ax1.set_title("(b) Three-draw identification", loc="left", fontweight="bold")
    ax1.grid(axis="x", color="#dddddd", linewidth=.7)

    fig.suptitle("Support and task density after deployment-state alignment",
                 y=1.03, fontsize=11, fontweight="bold")
    fig.tight_layout()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        kwargs = {"dpi": 240} if suffix == "png" else {}
        fig.savefig(args.output_dir / f"support_density_3draw.{suffix}",
                    bbox_inches="tight", **kwargs)


if __name__ == "__main__":
    main()
