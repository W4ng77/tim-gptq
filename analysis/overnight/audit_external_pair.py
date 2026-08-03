#!/usr/bin/env python3
"""Audit absolute WER and sample-level tails for a paired external ASR run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from rapidfuzz.distance import Levenshtein

from paired_external_bootstrap import normalize, read_rows


def compact(text: str, limit: int = 180) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uniform", type=Path, required=True)
    parser.add_argument("--task-density", type=Path, required=True)
    parser.add_argument("--dataset", action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=12)
    args = parser.parse_args()

    payload: dict[str, object] = {
        "uniform": str(args.uniform.resolve()),
        "task_density": str(args.task_density.resolve()),
        "datasets": {},
    }
    report = [
        "# External pair audit",
        "",
        "Absolute corpus WER and sample-level error tails. WER may exceed 100% when",
        "insertions dominate; lower task-density error is favorable.",
        "",
        "| Dataset | n | Uniform WER | Task-density WER | Delta | U max err/ref | K max err/ref |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    all_top: list[dict[str, object]] = []
    for dataset in args.dataset:
        left = read_rows(args.uniform / f"{dataset}.jsonl")
        right = read_rows(args.task_density / f"{dataset}.jsonl")
        if len(left) != len(right):
            raise ValueError(f"Row count mismatch for {dataset}")
        rows = []
        for a, b in zip(left, right, strict=True):
            if a["example_id"] != b["example_id"] or a["reference"] != b["reference"]:
                raise ValueError(f"Alignment mismatch for {dataset}/{a['example_id']}")
            ref = normalize(a["reference"]).split()
            if not ref:
                raise ValueError(f"Empty normalized reference for {dataset}/{a['example_id']}")
            u_pred = normalize(a["prediction"]).split()
            k_pred = normalize(b["prediction"]).split()
            u_err = Levenshtein.distance(ref, u_pred)
            k_err = Levenshtein.distance(ref, k_pred)
            rows.append({
                "dataset": dataset,
                "example_id": a["example_id"],
                "reference_words": len(ref),
                "uniform_prediction_words": len(u_pred),
                "task_density_prediction_words": len(k_pred),
                "uniform_errors": u_err,
                "task_density_errors": k_err,
                "error_delta": k_err - u_err,
                "reference": compact(a["reference"]),
                "uniform_prediction": compact(a["prediction"]),
                "task_density_prediction": compact(b["prediction"]),
            })
        lengths = np.asarray([row["reference_words"] for row in rows], dtype=np.int64)
        u_errors = np.asarray([row["uniform_errors"] for row in rows], dtype=np.int64)
        k_errors = np.asarray([row["task_density_errors"] for row in rows], dtype=np.int64)
        u_words = np.asarray([row["uniform_prediction_words"] for row in rows], dtype=np.int64)
        k_words = np.asarray([row["task_density_prediction_words"] for row in rows], dtype=np.int64)
        deltas = k_errors - u_errors
        u_wer = u_errors.sum() / lengths.sum()
        k_wer = k_errors.sum() / lengths.sum()
        capped_u_wer = np.minimum(u_errors, lengths).sum() / lengths.sum()
        capped_k_wer = np.minimum(k_errors, lengths).sum() / lengths.sum()
        stable = ((u_errors <= lengths) & (k_errors <= lengths)
                  & (u_words <= 3 * lengths) & (k_words <= 3 * lengths))
        stable_u_wer = u_errors[stable].sum() / lengths[stable].sum()
        stable_k_wer = k_errors[stable].sum() / lengths[stable].sum()
        top = sorted(rows, key=lambda row: (abs(row["error_delta"]), row["uniform_errors"]), reverse=True)[: args.top_k]
        all_top.extend(top)
        stat = {
            "n": len(rows),
            "reference_words": int(lengths.sum()),
            "uniform_errors": int(u_errors.sum()),
            "task_density_errors": int(k_errors.sum()),
            "uniform_wer": float(u_wer),
            "task_density_wer": float(k_wer),
            "delta": float(k_wer - u_wer),
            "capped_uniform_wer": float(capped_u_wer),
            "capped_task_density_wer": float(capped_k_wer),
            "capped_delta": float(capped_k_wer - capped_u_wer),
            "stable_pair_n": int(np.count_nonzero(stable)),
            "stable_pair_uniform_wer": float(stable_u_wer),
            "stable_pair_task_density_wer": float(stable_k_wer),
            "stable_pair_delta": float(stable_k_wer - stable_u_wer),
            "uniform_sample_error_rate_quantiles": {
                str(q): float(np.quantile(u_errors / lengths, q)) for q in (0.5, 0.9, 0.95, 0.99, 1.0)
            },
            "task_density_sample_error_rate_quantiles": {
                str(q): float(np.quantile(k_errors / lengths, q)) for q in (0.5, 0.9, 0.95, 0.99, 1.0)
            },
            "error_delta_quantiles": {
                str(q): float(np.quantile(deltas, q)) for q in (0.0, 0.01, 0.05, 0.5, 0.95, 0.99, 1.0)
            },
            "uniform_samples_over_100pct": int(np.count_nonzero(u_errors > lengths)),
            "task_density_samples_over_100pct": int(np.count_nonzero(k_errors > lengths)),
            "uniform_prediction_words_over_3x_ref": int(np.count_nonzero(
                u_words > 3 * lengths
            )),
            "task_density_prediction_words_over_3x_ref": int(np.count_nonzero(
                k_words > 3 * lengths
            )),
            "top_absolute_delta_examples": top,
        }
        payload["datasets"][dataset] = stat
        report.append(
            f"| {dataset} | {len(rows)} | {100*u_wer:.3f}% | {100*k_wer:.3f}% | "
            f"{100*(k_wer-u_wer):+.3f}pp | {(u_errors/lengths).max():.2f} | "
            f"{(k_errors/lengths).max():.2f} |"
        )

    report.extend([
        "",
        "## Tail-robust sensitivity",
        "",
        "Capped WER clips each utterance's edit count at its reference length. The",
        "stable-pair subset removes an utterance if either arm exceeds 100% WER or",
        "three times the reference length; it is diagnostic, not the primary metric.",
        "",
        "| Dataset | Capped U | Capped K | Capped delta | Stable n | Stable U | Stable K | Stable delta |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for dataset in args.dataset:
        stat = payload["datasets"][dataset]
        report.append(
            f"| {dataset} | {100*stat['capped_uniform_wer']:.3f}% | "
            f"{100*stat['capped_task_density_wer']:.3f}% | {100*stat['capped_delta']:+.3f}pp | "
            f"{stat['stable_pair_n']} | {100*stat['stable_pair_uniform_wer']:.3f}% | "
            f"{100*stat['stable_pair_task_density_wer']:.3f}% | "
            f"{100*stat['stable_pair_delta']:+.3f}pp |"
        )

    report.extend(["", "## Largest paired error changes", ""])
    for row in sorted(all_top, key=lambda item: abs(item["error_delta"]), reverse=True)[: args.top_k]:
        report.extend([
            f"### {row['dataset']} / {row['example_id']}",
            "",
            f"Errors U/K/delta: `{row['uniform_errors']}/{row['task_density_errors']}/{row['error_delta']:+d}`; "
            f"words ref/U/K: `{row['reference_words']}/{row['uniform_prediction_words']}/{row['task_density_prediction_words']}`.",
            "",
            f"- Reference: {row['reference']}",
            f"- Uniform: {row['uniform_prediction']}",
            f"- Task density: {row['task_density_prediction']}",
            "",
        ])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "external_pair_audit.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "EXTERNAL_PAIR_AUDIT.md").write_text("\n".join(report), encoding="utf-8")


if __name__ == "__main__":
    main()
