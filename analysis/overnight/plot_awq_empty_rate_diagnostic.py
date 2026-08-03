#!/usr/bin/env python3
"""Plot terminal Qwen deployment-AWQ empty rate against capped WER excess."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from crossed_external_bootstrap import load_pair  # noqa: E402

FLEURS = [
    "fleurs-en-us", "fleurs-de-de", "fleurs-fr-fr", "fleurs-es-419",
    "fleurs-pt-br",
]


def run(name: str) -> Path:
    return ROOT / "runs" / name


PANELS = [
    {
        "model": "0.6B", "target": "FLEURS", "datasets": FLEURS,
        "baseline": [
            "w2gf-external/qwen06-text-backbone-w3-uniform-calseed20260729-fleurs5-official-v1",
            "w2gf-external-draw30/qwen06-text-backbone-w3-uniform-calseed20260730-fleurs5-official-v1",
            "w2gf-fleurs-draw31/qwen06-text-backbone-w3-uniform-calseed20260731-fleurs5-official-v1",
        ],
        "candidate": [
            f"w2gf-qwen06-fleurs-awq-deploy/qwen06-text-backbone-w3-awq-deployseq-calseed{s}-fleurs5-official-v1"
            for s in (20260729, 20260730, 20260731)
        ],
    },
    {
        "model": "0.6B", "target": "ProfASR", "datasets": ["profasr-no-prompt"],
        "baseline": [
            "w2gf-profasr-v2/qwen06-text-backbone-w3-uniform-calseed20260729-profasr-official-v2",
            "w2gf-profasr-v2/qwen06-text-backbone-w3-uniform-calseed20260730-profasr-official-v2",
            "w2gf-profasr-v2-draw31/qwen06-text-backbone-w3-uniform-calseed20260731-profasr-official-v2",
        ],
        "candidate": [
            f"w2gf-qwen06-profasr-awq-deploy/qwen06-text-backbone-w3-awq-deployseq-calseed{s}-profasr-official-v2"
            for s in (20260729, 20260730, 20260731)
        ],
    },
    {
        "model": "0.6B", "target": "ContextASR", "datasets": ["contextasr-speech-en"],
        "baseline": [
            "w2gf-external/qwen06-text-backbone-w3-uniform-calseed20260729-contextasr-official-v1",
            "w2gf-external-draw30/qwen06-text-backbone-w3-uniform-calseed20260730-contextasr-official-v1",
            "w2gf-qwen06-context31/qwen06-text-backbone-w3-uniform-calseed20260731-contextasr-official-v1",
        ],
        "candidate": [
            f"w2gf-qwen06-contextasr-awq-deploy/qwen06-text-backbone-w3-awq-deployseq-calseed{s}-contextasr-speech-en-official-v1"
            for s in (20260729, 20260730, 20260731)
        ],
    },
    {
        "model": "1.7B", "target": "FLEURS", "datasets": FLEURS,
        "baseline": [
            "w2gf-qwen17-external/qwen17-text-backbone-w3-uniform-calseed20260729-fleurs5-official-v1",
            "w2gf-qwen17-external-draw30/qwen17-text-backbone-w3-uniform-calseed20260730-fleurs5-official-v1",
            "w2gf-fleurs-draw31/qwen17-text-backbone-w3-uniform-calseed20260731-fleurs5-official-v1",
        ],
        "candidate": [
            f"w2gf-qwen17-fleurs-awq-deploy/qwen17-text-backbone-w3-awq-deployseq-calseed{s}-fleurs5-official-v1"
            for s in (20260729, 20260730, 20260731)
        ],
    },
    {
        "model": "1.7B", "target": "ProfASR", "datasets": ["profasr-no-prompt"],
        "baseline": [
            "w2gf-profasr-v2/qwen17-text-backbone-w3-uniform-calseed20260729-profasr-official-v2",
            "w2gf-profasr-v2-draw30/qwen17-text-backbone-w3-uniform-calseed20260730-profasr-official-v2",
            "w2gf-profasr-v2-draw31/qwen17-text-backbone-w3-uniform-calseed20260731-profasr-official-v2",
        ],
        "candidate": [
            f"w2gf-qwen17-profasr-awq-deploy/qwen17-text-backbone-w3-awq-deployseq-calseed{s}-profasr-official-v2"
            for s in (20260729, 20260730, 20260731)
        ],
    },
    {
        "model": "1.7B", "target": "ContextASR", "datasets": ["contextasr-speech-en"],
        "baseline": [
            f"w2gf-qwen17-context{suffix}/qwen17-text-backbone-w3-uniform-calseed{s}-contextasr-official-v1"
            for suffix, s in (("", 20260729), ("30", 20260730), ("31", 20260731))
        ],
        "candidate": [
            f"w2gf-qwen17-contextasr-awq-deploy/qwen17-text-backbone-w3-awq-deployseq-calseed{s}-contextasr-speech-en-official-v1"
            for s in (20260729, 20260730, 20260731)
        ],
    },
]


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = (start + stop - 1) / 2
        start = stop
    return ranks


def main() -> None:
    points: list[dict] = []
    for panel in PANELS:
        for draw, (baseline_name, candidate_name) in enumerate(
            zip(panel["baseline"], panel["candidate"], strict=True), start=29
        ):
            baseline, candidate = run(baseline_name), run(candidate_name)
            for directory in (baseline, candidate):
                state = json.loads((directory / "status.json").read_text())["state"]
                if state != "completed":
                    raise RuntimeError(f"non-terminal input: {directory}")
            deltas, empty, total = [], 0, 0
            for dataset in panel["datasets"]:
                lengths, base_errors, candidate_errors = load_pair(
                    baseline, candidate, dataset, True
                )
                deltas.append(
                    100 * (candidate_errors.sum() - base_errors.sum()) / lengths.sum()
                )
                rows = read_jsonl(candidate / f"{dataset}.jsonl")
                empty += sum(not str(row.get("prediction", "")).strip() for row in rows)
                total += len(rows)
            points.append({
                "model": panel["model"], "target": panel["target"], "draw": draw,
                "empty_rate_percent": 100 * empty / total,
                "capped_wer_excess_pp": float(np.mean(deltas)),
                "empty": empty, "examples": total,
            })

    x = np.asarray([row["empty_rate_percent"] for row in points])
    y = np.asarray([row["capped_wer_excess_pp"] for row in points])
    pearson = float(np.corrcoef(x, y)[0, 1])
    spearman = float(np.corrcoef(average_ranks(x), average_ranks(y))[0, 1])
    minimum_empty = min(points, key=lambda row: row["empty_rate_percent"])
    payload = {
        "diagnostic_only": True,
        "points": points,
        "pearson": pearson,
        "spearman": spearman,
        "minimum_empty_rate_point": minimum_empty,
    }
    (HERE / "awq_empty_rate_diagnostic.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )

    colors = {"FLEURS": "#4477AA", "ProfASR": "#228833", "ContextASR": "#CC6677"}
    markers = {"0.6B": "o", "1.7B": "s"}
    fig, ax = plt.subplots(figsize=(7.2, 4.1))
    for target, color in colors.items():
        for model, marker in markers.items():
            subset = [p for p in points if p["target"] == target and p["model"] == model]
            ax.scatter(
                [p["empty_rate_percent"] for p in subset],
                [p["capped_wer_excess_pp"] for p in subset],
                s=58, marker=marker, color=color, edgecolor="white", linewidth=.7,
                label=f"{model} / {target}", zorder=3,
            )
    ax.axhline(0, color="#333333", lw=.8)
    ax.annotate(
        f"minimum empty rate: {minimum_empty['empty_rate_percent']:.2f}%\n"
        f"still {minimum_empty['capped_wer_excess_pp']:+.1f} pp",
        (minimum_empty["empty_rate_percent"], minimum_empty["capped_wer_excess_pp"]),
        xytext=(16, 12), textcoords="offset points", fontsize=8,
        arrowprops={"arrowstyle": "->", "lw": .8, "color": "#555555"},
    )
    ax.set_xlabel("Empty prediction rate (%)")
    ax.set_ylabel("Capped WER excess over matched uniform GPTQ (pp)")
    ax.set_title("Empty-output rate is not a sufficient solver-health diagnostic")
    ax.grid(axis="both", color="#DDDDDD", lw=.6, alpha=.8)
    ax.legend(ncol=2, fontsize=7.5, frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(HERE / "figures" / "awq_empty_rate_diagnostic.pdf", bbox_inches="tight")
    fig.savefig(HERE / "figures" / "awq_empty_rate_diagnostic.png", dpi=220,
                bbox_inches="tight")
    plt.close(fig)

    (HERE / "AWQ_EMPTY_RATE_DIAGNOSTIC.md").write_text(
        "# AWQ empty-rate diagnostic\n\n"
        f"Across {len(points)} terminal Qwen3-ASR deployment-AWQ draw×target cells, "
        f"empty rate and capped excess WER have Pearson `{pearson:+.3f}` and "
        f"Spearman `{spearman:+.3f}` association. Empty rate therefore carries "
        f"some failure-severity information, but the minimum-empty cell has only "
        f"`{minimum_empty['empty_rate_percent']:.2f}%` empty outputs and still "
        f"`{minimum_empty['capped_wer_excess_pp']:+.3f}pp` capped excess WER. "
        "Therefore empty-output rate detects one failure phenotype but cannot "
        "certify a usable solver/rollout basin. This is a descriptive diagnostic, "
        "not a fitted router.\n\n"
        "[Figure](figures/awq_empty_rate_diagnostic.pdf) | "
        "[Frozen points](awq_empty_rate_diagnostic.json)\n"
    )


if __name__ == "__main__":
    main()
