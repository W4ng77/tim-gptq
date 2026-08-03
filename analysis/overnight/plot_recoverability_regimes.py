#!/usr/bin/env python3
"""Plot task-measure gain against exact-target FP16 headroom."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def rows(path: Path) -> dict[str, dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {row["label"]: row for row in payload["methods"]}


def point(name: str, family: str, bits: int, fp16: float, uniform: float,
          task: float, source: str) -> dict:
    return {
        "name": name,
        "family": family,
        "bits": bits,
        "fp16_wer": fp16,
        "uniform_wer": uniform,
        "task_wer": task,
        "uniform_excess_over_fp16": uniform - fp16,
        "task_measure_gain": uniform - task,
        "source": source,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.analysis_root

    large_w2_path = root / "large_fleurs_method_summary_terminal/fleurs_method_summary.json"
    large_w3_path = root / "large_fleurs_w3_terminal_summary/fleurs_method_summary.json"
    large_w4_path = root / "large_fleurs_w4_terminal_summary/fleurs_method_summary.json"
    qwen_path = root / "qwen06_fleurs_method_summary_terminal/fleurs_method_summary.json"
    qwen06_prof_path = root / "qwen06_profasr_v2_crossed_3draw/crossed_external_bootstrap.json"
    qwen06_prof_fp16_path = (args.repo_root / "runs/w2gf-qwen06-profasr-fp16/"
                             "qwen06-fp16-profasr-official-v2/metrics.json")
    qwen17_fleurs_path = root / "qwen17_fleurs5_crossed_3draw/crossed_external_bootstrap.json"
    qwen17_fleurs_fp16_path = (args.repo_root / "runs/w2gf-qwen17-fleurs-fp16/"
                               "qwen17-fp16-fleurs5-official-v1/metrics.json")
    qwen17_prof_path = root / "qwen17_profasr_v2_crossed_3draw/crossed_external_bootstrap.json"
    qwen17_prof_fp16_path = (args.repo_root / "runs/w2gf-qwen17-profasr-fp16/"
                             "qwen17-fp16-profasr-official-v2/metrics.json")
    vox_fleurs_path = root / "voxtral_fleurs_fp16_headroom_terminal/fleurs_method_summary.json"
    prof_path = root / "voxtral_profasr_v2_aligned_vs_uniform_3draw/crossed_external_contrast.json"
    prof_w3_path = root / "voxtral_profasr_w3_terminal_raw/crossed_external_contrast.json"
    prof_fp16_path = (args.repo_root / "runs/w2gf-voxtral-profasr-baselines-v2/"
                      "voxtral-mini-fp16-profasr-official-v2/metrics.json")
    vox_context_path = root / "voxtral_contextasr_crossed_3draw/crossed_external_contrast.json"
    vox_context_fp16_path = (args.repo_root / "runs/w2gf-voxtral-contextasr-fp16/"
                             "voxtral-mini-fp16-contextasr-speech-en-official-v1/metrics.json")
    large_prof_path = root / "large_profasr_v2_crossed_3draw/crossed_external_contrast.json"
    large_prof_fp16_path = (args.repo_root / "runs/w2gf-large-profasr-fp16/"
                            "large-fp16-profasr-official-v2/metrics.json")

    large_w2, large_w3, large_w4 = rows(large_w2_path), rows(large_w3_path), rows(large_w4_path)
    qwen, vox_fleurs = rows(qwen_path), rows(vox_fleurs_path)
    qwen06_prof = json.loads(qwen06_prof_path.read_text(encoding="utf-8"))["macro"]
    qwen06_prof_fp16 = json.loads(
        qwen06_prof_fp16_path.read_text(encoding="utf-8")
    )["evaluations"][0]["wer"]
    qwen17_fleurs = json.loads(qwen17_fleurs_path.read_text(encoding="utf-8"))["macro"]
    qwen17_fleurs_fp16 = sum(
        row["wer"] for row in json.loads(
            qwen17_fleurs_fp16_path.read_text(encoding="utf-8")
        )["evaluations"]
    ) / 5
    qwen17_prof = json.loads(qwen17_prof_path.read_text(encoding="utf-8"))["macro"]
    qwen17_prof_fp16 = json.loads(
        qwen17_prof_fp16_path.read_text(encoding="utf-8")
    )["evaluations"][0]["wer"]
    large_fp16 = large_w2["fp16"]["raw_macro_draw_mean"]
    prof = json.loads(prof_path.read_text(encoding="utf-8"))["macro"]
    prof_w3 = json.loads(prof_w3_path.read_text(encoding="utf-8"))["macro"]
    prof_fp16 = json.loads(prof_fp16_path.read_text(encoding="utf-8"))["evaluations"][0]["wer"]
    vox_context = json.loads(vox_context_path.read_text(encoding="utf-8"))["macro"]
    vox_context_fp16 = json.loads(
        vox_context_fp16_path.read_text(encoding="utf-8")
    )["evaluations"][0]["wer"]
    large_prof = json.loads(large_prof_path.read_text(encoding="utf-8"))["macro"]
    large_prof_fp16 = json.loads(
        large_prof_fp16_path.read_text(encoding="utf-8")
    )["evaluations"][0]["wer"]

    points = [
        point("Large/FLEURS W2", "Large/FLEURS", 2, large_fp16,
              large_w2["uniform"]["raw_macro_draw_mean"],
              large_w2["task"]["raw_macro_draw_mean"], str(large_w2_path)),
        point("Large/FLEURS W3", "Large/FLEURS", 3, large_fp16,
              large_w3["uniform_w3"]["raw_macro_draw_mean"],
              large_w3["task_w3"]["raw_macro_draw_mean"], str(large_w3_path)),
        point("Large/FLEURS W4", "Large/FLEURS", 4, large_fp16,
              large_w4["uniform_w4"]["raw_macro_draw_mean"],
              large_w4["task_w4"]["raw_macro_draw_mean"], str(large_w4_path)),
        point("Qwen06/FLEURS W3", "Qwen06/FLEURS", 3,
              qwen["fp16"]["raw_macro_draw_mean"],
              qwen["uniform"]["raw_macro_draw_mean"],
              qwen["task"]["raw_macro_draw_mean"], str(qwen_path)),
        point("Qwen06/ProfASR W3", "Qwen06/ProfASR", 3, qwen06_prof_fp16,
              qwen06_prof["uniform_wer_draw_mean"],
              qwen06_prof["task_density_wer_draw_mean"], str(qwen06_prof_path)),
        point("Qwen17/FLEURS W3", "Qwen17/FLEURS", 3, qwen17_fleurs_fp16,
              qwen17_fleurs["uniform_wer_draw_mean"],
              qwen17_fleurs["task_density_wer_draw_mean"], str(qwen17_fleurs_path)),
        point("Qwen17/ProfASR W3", "Qwen17/ProfASR", 3, qwen17_prof_fp16,
              qwen17_prof["uniform_wer_draw_mean"],
              qwen17_prof["task_density_wer_draw_mean"], str(qwen17_prof_path)),
        point("Voxtral/FLEURS W2", "Voxtral/FLEURS", 2,
              vox_fleurs["fp16"]["raw_macro_draw_mean"],
              vox_fleurs["uniform_w2"]["raw_macro_draw_mean"],
              vox_fleurs["task_w2"]["raw_macro_draw_mean"], str(vox_fleurs_path)),
        point("Voxtral/ProfASR W2", "Voxtral/ProfASR", 2, prof_fp16,
              prof["baseline_wer_draw_mean"], prof["candidate_wer_draw_mean"],
              str(prof_path)),
        point("Voxtral/ProfASR W3", "Voxtral/ProfASR", 3, prof_fp16,
              prof_w3["baseline_wer_draw_mean"], prof_w3["candidate_wer_draw_mean"],
              str(prof_w3_path)),
        point("Voxtral/Context W2", "Voxtral/Context", 2, vox_context_fp16,
              vox_context["baseline_wer_draw_mean"],
              vox_context["candidate_wer_draw_mean"], str(vox_context_path)),
        point("Large/ProfASR W2", "Large/ProfASR", 2, large_prof_fp16,
              large_prof["baseline_wer_draw_mean"],
              large_prof["candidate_wer_draw_mean"], str(large_prof_path)),
    ]

    colors = {
        "Large/FLEURS": "#3568a8",
        "Qwen06/FLEURS": "#5a9f68",
        "Qwen06/ProfASR": "#4c9f9a",
        "Qwen17/FLEURS": "#b279a2",
        "Qwen17/ProfASR": "#59a14f",
        "Voxtral/FLEURS": "#c44e52",
        "Voxtral/ProfASR": "#e08b3e",
        "Voxtral/Context": "#d4a72c",
        "Large/ProfASR": "#8a63a8",
    }
    markers = {2: "o", 3: "s", 4: "^"}
    offsets = {
        "Large/FLEURS W2": (-2, 9),
        "Large/FLEURS W3": (7, 11),
        "Large/FLEURS W4": (7, -19),
        "Qwen06/FLEURS W3": (-92, -21),
        "Qwen06/ProfASR W3": (8, 9),
        "Qwen17/FLEURS W3": (8, -22),
        "Qwen17/ProfASR W3": (-8, 20),
        "Voxtral/FLEURS W2": (-90, -18),
        "Voxtral/ProfASR W2": (8, 8),
        "Voxtral/ProfASR W3": (8, 25),
        "Voxtral/Context W2": (8, 18),
        "Large/ProfASR W2": (8, -17),
    }

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
    })
    fig, ax = plt.subplots(figsize=(7.4, 4.5), constrained_layout=True)
    ax.axvspan(-0.25, 0.5, color="#777777", alpha=0.09, label="Near-FP16 saturation")
    ax.axhline(0, color="#555555", linewidth=1.1, linestyle="--")
    for row in points:
        x = 100 * row["uniform_excess_over_fp16"]
        y = 100 * row["task_measure_gain"]
        ax.scatter(x, y, s=82, color=colors[row["family"]], marker=markers[row["bits"]],
                   edgecolor="white", linewidth=0.8, zorder=3)
        if x >= 0.5 or row["name"] == "Large/ProfASR W2":
            dx, dy = offsets[row["name"]]
            horizontal_alignment = (
                "right" if row["name"] == "Qwen17/ProfASR W3" else "left"
            )
            ax.annotate(row["name"], (x, y), xytext=(dx, dy), textcoords="offset points",
                        fontsize=8.0, color="#252525", ha=horizontal_alignment)
    ax.text(10.7, 0.28, "benefit", ha="center", va="bottom", color="#3a7d44", fontsize=9)
    ax.text(10.7, -0.28, "harm", ha="center", va="top", color="#a23a3d", fontsize=9)
    ax.set_xlim(-0.25, 17.25)
    ax.set_ylim(-5.2, 6.1)
    ax.set_xlabel("Uniform GPTQ excess over exact-target FP16 (WER pp)")
    ax.set_ylabel("Task-measure gain over uniform GPTQ (WER pp)")
    ax.set_title("Headroom is necessary but not sufficient", loc="left", fontweight="bold")
    ax.grid(alpha=0.22, linewidth=0.7)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    legend_handles = [
        plt.Line2D([], [], linestyle="", marker=markers[bits], color="#555555",
                   markerfacecolor="#777777", markersize=7, label=f"W{bits}")
        for bits in (2, 3, 4)
    ]
    ax.legend(handles=legend_handles, loc="upper left", ncol=3, frameon=False)

    inset = ax.inset_axes([0.23, 0.59, 0.27, 0.25])
    inset.axhline(0, color="#666666", linewidth=0.8, linestyle="--")
    inset_offsets = {
        "Large/FLEURS W4": (3, -13),
        "Large/FLEURS W3": (-18, -14),
        "Voxtral/ProfASR W3": (-62, 4),
        "Large/ProfASR W2": (-5, -14),
    }
    for row in points:
        x = 100 * row["uniform_excess_over_fp16"]
        y = 100 * row["task_measure_gain"]
        if x >= 0.5 or abs(y) > 0.04:
            continue
        inset.scatter(x, y, s=42, color=colors[row["family"]],
                      marker=markers[row["bits"]], edgecolor="white",
                      linewidth=0.6, zorder=3)
        dx, dy = inset_offsets[row["name"]]
        inset.annotate(row["name"].replace("/FLEURS", "").replace("/ProfASR", "/Prof"),
                       (x, y), xytext=(dx, dy), textcoords="offset points",
                       fontsize=6.7, color="#252525")
    inset.set_xlim(-0.01, 0.50)
    inset.set_ylim(-0.032, 0.032)
    inset.set_xticks([0.0, 0.2, 0.4])
    inset.set_yticks([-0.02, 0.0, 0.02])
    inset.tick_params(labelsize=6.5, length=2)
    inset.set_title("Saturation zoom", fontsize=7.5, loc="left", pad=2)
    inset.grid(alpha=0.18, linewidth=0.5)
    for spine in ("top", "right"):
        inset.spines[spine].set_visible(False)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "recoverability_regime_points.json").write_text(
        json.dumps(points, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    for suffix in ("png", "pdf"):
        fig.savefig(args.output_dir / f"recoverability_regimes.{suffix}",
                    dpi=220, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
