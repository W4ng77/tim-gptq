#!/usr/bin/env python3
"""Conceptual paper figure for the map-support-density-geometry decomposition."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def box(ax, xy, width, height, text, face, edge="#555555", fontsize=8.5,
        linewidth=1.0, weight="normal"):
    patch = FancyBboxPatch(
        xy, width, height, boxstyle="round,pad=0.018,rounding_size=0.018",
        facecolor=face, edgecolor=edge, linewidth=linewidth)
    ax.add_patch(patch)
    ax.text(xy[0] + width / 2, xy[1] + height / 2, text, ha="center", va="center",
            fontsize=fontsize, fontweight=weight, linespacing=1.25)
    return patch


def arrow(ax, start, end, text=None, color="#555555", rad=0.0, text_offset=(0, 0)):
    patch = FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=11,
                            linewidth=1.15, color=color,
                            connectionstyle=f"arc3,rad={rad}")
    ax.add_patch(patch)
    if text:
        ax.text((start[0] + end[0]) / 2 + text_offset[0],
                (start[1] + end[1]) / 2 + text_offset[1], text,
                ha="center", va="center", fontsize=7.5, color=color,
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.0})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
    fig, ax = plt.subplots(figsize=(10.5, 5.4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    blue = "#DCEAF7"
    red = "#F9DEDC"
    green = "#DDEEDB"
    gold = "#F8E8C6"
    gray = "#EEF0F2"
    purple = "#E8E0F3"

    ax.text(.015, .925, "Target deployment measure", fontsize=10.5,
            fontweight="bold", color="#8B1A1A")
    box(ax, (.015, .75), .17, .115, "$q_{dep}=(a,p_{dep},y_{<t})$\n"
        "deployment tuples", red)
    box(ax, (.245, .75), .16, .115, "$z=\\phi_\\ell(q_{dep})$\n"
        "consumed states", red)
    box(ax, (.465, .75), .17, .115, "$d\\mu_\\ell=\\kappa_\\ell\\,d\\rho_{dep}$\n"
        "task measure", red, weight="bold")
    box(ax, (.695, .75), .15, .115, "$H_\\ell^*=\\int zz^\\top d\\mu_\\ell$\n"
        "ideal Gram", red)
    box(ax, (.89, .75), .095, .115, "local task\nloss", red)
    arrow(ax, (.185, .807), (.245, .807), "$\\phi_{p_{dep}}$")
    arrow(ax, (.405, .807), (.465, .807), "pullback $\\kappa$")
    arrow(ax, (.635, .807), (.695, .807), "second moment")
    arrow(ax, (.845, .807), (.89, .807))

    ax.text(.015, .59, "Calibration approximation", fontsize=10.5,
            fontweight="bold", color="#174A76")
    box(ax, (.015, .415), .17, .115, "$q_{cal}=(a,p_{cal},S)$\n"
        "calibration tuples", blue)
    box(ax, (.245, .415), .16, .115, "$z=\\phi_\\ell(q_{cal})$\n"
        "observed states", blue)
    box(ax, (.465, .415), .17, .115, "$\\nu_\\ell^{p_{cal},S}$\n"
        "occupancy measure", blue)
    box(ax, (.695, .415), .15, .115, "$\\widehat H_\\ell=\\sum_t w_tz_tz_t^\\top$\n"
        "bounded Gram", blue)
    box(ax, (.89, .415), .095, .115, "groupwise\nGPTQ", blue)
    arrow(ax, (.185, .472), (.245, .472), "$\\phi_{p_{cal}}$")
    arrow(ax, (.405, .472), (.465, .472), "retain rows $S$")
    arrow(ax, (.635, .472), (.695, .472), "$w=\\Pi_{KL}(d\\mu/d\\nu)$")
    arrow(ax, (.845, .472), (.89, .472))

    # Ordered discrepancies linking the calibration object to the target.
    x_positions = (.325, .55, .77, .935)
    labels = (
        "1  Map / interface\n$p_{cal}\\ne p_{dep}$",
        "2  Missing support\n$\\mu_{\\perp}$",
        "3  Density mismatch\n$r=d\\mu/d\\nu$",
        "4  Solver geometry\ninverse sensitivity",
    )
    colors = (purple, gold, green, gray)
    widths = (.17, .16, .16, .115)
    for x, label, color, width in zip(x_positions, labels, colors, widths):
        box(ax, (x - width / 2, .595), width, .085, label, color,
            fontsize=7.7, linewidth=.9, weight="bold")
        arrow(ax, (x, .68), (x, .735), color="#888888")
        arrow(ax, (x, .545), (x, .595), color="#888888")

    ax.text(.015, .325, "Identification: change one link while preserving the others",
            fontsize=10.5, fontweight="bold")
    evidence = (
        ("Template x support x density", "same model/data/seed", "map-conditioned support", purple),
        ("Padding/sink intervention", "valid weights identical", "excluded consumed states", gold),
        ("Within-sample permutation", "histogram/mean/ESS fixed", "task-state covariance", green),
        ("RHT factorial", "task measure fixed", "solver amplification", gray),
    )
    x0 = (.015, .265, .515, .765)
    for x, (title, control, claim, color) in zip(x0, evidence):
        box(ax, (x, .105), .22, .145,
            f"{title}\n{control}\n$\\Rightarrow$ {claim}", color,
            fontsize=8.0, weight="normal")

    ax.text(.5, .035,
            "Principle: align the deployment state map before support completion; estimate task density only on shared support.",
            ha="center", va="center", fontsize=10, fontweight="bold", color="#222222")
    fig.suptitle("Decoder-induced calibration measures for ASR PTQ", y=.985,
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        kwargs = {"dpi": 240} if suffix == "png" else {}
        fig.savefig(args.output_dir / f"measure_framework.{suffix}",
                    bbox_inches="tight", **kwargs)
    print(args.output_dir / "measure_framework.pdf")


if __name__ == "__main__":
    main()
