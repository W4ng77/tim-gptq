#!/usr/bin/env python3
"""Plot permutation identification internally and on official external targets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load(path: Path) -> np.ndarray:
    row = json.loads(path.read_text())["macro"]
    return 100 * np.asarray([row["estimate"], row["ci95_lower"], row["ci95_upper"]])


def load_nested(path: Path, contrast: str) -> np.ndarray:
    row = json.loads(path.read_text())["bootstrap"][contrast]["macro"]
    return 100 * np.asarray([row["estimate"], row["ci95_lower"], row["ci95_upper"]])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case", nargs=3, action="append", default=[],
        metavar=("LABEL", "PERM_MINUS_ALIGNED_JSON", "PERM_MINUS_UNIFORM_JSON"),
    )
    parser.add_argument(
        "--nested-case", nargs=2, action="append", default=[],
        metavar=("LABEL", "NESTED_BOOTSTRAP_JSON"),
        help="add a nested calibration-draw x permutation-seed result",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-name", default="alignment_internal_external")
    args = parser.parse_args()
    if not args.case and not args.nested_case:
        parser.error("at least one --case or --nested-case is required")
    labels = [row[0].replace("\\n", "\n") for row in args.nested_case + args.case]
    aligned = np.stack(
        [load_nested(Path(row[1]), "permuted_minus_aligned") for row in args.nested_case]
        + [load(Path(row[1])) for row in args.case]
    )
    uniform = np.stack(
        [load_nested(Path(row[1]), "permuted_minus_uniform") for row in args.nested_case]
        + [load(Path(row[2])) for row in args.case]
    )

    plt.rcParams.update({"font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False})
    # Keep six-row manuscript panels within the ACL body-page budget while
    # retaining enough vertical separation for paired contrasts.
    fig_height = max(3.4, 0.35 * len(labels) + 1.20)
    fig, ax = plt.subplots(figsize=(7.5, fig_height))
    y = np.arange(len(labels))[::-1]
    ax.axvline(0, color="#777777", linewidth=.9)
    for yi, rows, color, marker, label in (
        (y + .11, aligned, "#B22222", "o", "Permuted − aligned"),
        (y - .11, uniform, "#777777", "s", "Permuted − uniform"),
    ):
        ax.errorbar(
            rows[:, 0], yi,
            xerr=[rows[:, 0] - rows[:, 1], rows[:, 2] - rows[:, 0]],
            fmt=marker, color=color, ecolor=color, capsize=3,
            linewidth=1.25, markersize=5.5, label=label,
        )
    ax.set_yticks(y, labels)
    ax.set_xlabel("Permutation-induced WER change (pp; higher is worse)")
    ax.set_title("Weight–state correspondence, not weight marginals, drives the gain",
                 loc="left", fontsize=11, fontweight="bold")
    ax.grid(axis="x", color="#dddddd", linewidth=.7)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        kwargs = {"dpi": 240} if suffix == "png" else {}
        fig.savefig(args.output_dir / f"{args.output_name}.{suffix}",
                    bbox_inches="tight", **kwargs)


if __name__ == "__main__":
    main()
