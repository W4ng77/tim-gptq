#!/usr/bin/env python3
"""Conservative calibration-only selector for task-measure PTQ candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from rapidfuzz.distance import Levenshtein

from trajectory_stability import analyze, load_score


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fp16-run", type=Path, required=True)
    parser.add_argument("--uniform-run", type=Path, required=True)
    parser.add_argument("--task-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reps", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260731)
    return parser.parse_args()


def paired_rows(fp16_rows, uniform_rows, task_rows):
    ids = list(fp16_rows)
    if set(ids) != set(uniform_rows) or set(ids) != set(task_rows):
        raise ValueError("FP16, uniform, and task example ids differ")
    rows = []
    for example_id in ids:
        reference, fp16_prediction = fp16_rows[example_id]
        uniform_reference, uniform_prediction = uniform_rows[example_id]
        task_reference, task_prediction = task_rows[example_id]
        if reference != uniform_reference or reference != task_reference:
            raise ValueError(f"Reference mismatch for {example_id}")
        ref = reference.split()
        fp16 = fp16_prediction.split()
        uniform = uniform_prediction.split()
        task = task_prediction.split()
        rows.append(
            {
                "reference_words": max(len(ref), 1),
                "fp16_words": max(len(fp16), 1),
                "fp16_errors": Levenshtein.distance(ref, fp16),
                "uniform_errors": Levenshtein.distance(ref, uniform),
                "task_errors": Levenshtein.distance(ref, task),
                "uniform_trajectory_edits": Levenshtein.distance(fp16, uniform),
                "task_trajectory_edits": Levenshtein.distance(fp16, task),
            }
        )
    return rows


def ratio_delta(rows, candidate_key, baseline_key, denominator_key, indices=None):
    if indices is None:
        selected = rows
    else:
        selected = (rows[index] for index in indices)
    candidate = 0
    baseline = 0
    denominator = 0
    for row in selected:
        candidate += row[candidate_key]
        baseline += row[baseline_key]
        denominator += row[denominator_key]
    return (candidate - baseline) / max(denominator, 1)


def bootstrap_delta(
    rows,
    candidate_key,
    baseline_key,
    denominator_key,
    reps,
    rng,
):
    estimate = ratio_delta(
        rows,
        candidate_key,
        baseline_key,
        denominator_key,
    )
    n = len(rows)
    samples = np.empty(reps, dtype=np.float64)
    for rep in range(reps):
        indices = rng.integers(0, n, size=n)
        samples[rep] = ratio_delta(
            rows,
            candidate_key,
            baseline_key,
            denominator_key,
            indices,
        )
    lower, upper = np.quantile(samples, [0.025, 0.975])
    return {
        "estimate": float(estimate),
        "ci95_lower": float(lower),
        "ci95_upper": float(upper),
    }


def main():
    args = parse_args()
    fp16_rows = load_score(args.fp16_run)
    uniform_rows = load_score(args.uniform_run)
    task_rows = load_score(args.task_run)
    rows = paired_rows(fp16_rows, uniform_rows, task_rows)
    rng = np.random.default_rng(args.seed)

    headroom = bootstrap_delta(
        rows,
        "uniform_errors",
        "fp16_errors",
        "reference_words",
        args.reps,
        rng,
    )
    task_wer = bootstrap_delta(
        rows,
        "task_errors",
        "uniform_errors",
        "reference_words",
        args.reps,
        rng,
    )
    task_trajectory = bootstrap_delta(
        rows,
        "task_trajectory_edits",
        "uniform_trajectory_edits",
        "fp16_words",
        args.reps,
        rng,
    )
    uniform_stability = analyze(fp16_rows, uniform_rows)
    task_stability = analyze(fp16_rows, task_rows)

    baseline_feasible = bool(uniform_stability["stability_gate_pass"])
    recoverable_headroom = headroom["ci95_lower"] > 0.0
    task_identifiable = (
        task_wer["ci95_upper"] < 0.0
        and task_trajectory["ci95_upper"] < 0.0
    )
    select_task = (
        baseline_feasible
        and bool(task_stability["stability_gate_pass"])
        and recoverable_headroom
        and task_identifiable
    )
    payload = {
        "protocol": {
            "baseline_feasible": "uniform collapse Wilson UCB95 <= 5%",
            "recoverable_headroom": (
                "paired bootstrap CI95 lower bound for "
                "uniform-minus-FP16 held-out WER > 0"
            ),
            "task_identifiable": (
                "paired bootstrap CI95 upper bounds for task-minus-uniform "
                "held-out WER and FP16-relative trajectory edit are both < 0"
            ),
            "selection": (
                "select task only if uniform and task are stable, recoverable "
                "headroom is certified, and task is identifiable"
            ),
        },
        "sources": {
            "fp16": str(args.fp16_run.resolve()),
            "uniform": str(args.uniform_run.resolve()),
            "task": str(args.task_run.resolve()),
        },
        "num_examples": len(rows),
        "uniform_stability": uniform_stability,
        "task_stability": task_stability,
        "uniform_minus_fp16_wer": headroom,
        "task_minus_uniform_wer": task_wer,
        "task_minus_uniform_trajectory_edit": task_trajectory,
        "decision": {
            "baseline_feasible": baseline_feasible,
            "task_feasible": bool(task_stability["stability_gate_pass"]),
            "recoverable_headroom": recoverable_headroom,
            "task_identifiable": task_identifiable,
            "select_task": select_task,
        },
        "reps": args.reps,
        "seed": args.seed,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "trajectory_candidate_selector.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    def ci(row):
        return (
            f"{100*row['estimate']:+.3f} "
            f"[{100*row['ci95_lower']:+.3f},"
            f"{100*row['ci95_upper']:+.3f}]"
        )

    decision = payload["decision"]
    lines = [
        "# Conservative trajectory candidate selector",
        "",
        f"Examples: `{len(rows)}`; paired bootstrap replicates: `{args.reps}`.",
        "",
        "| Diagnostic | Δ pp [95% CI] | Pass |",
        "|---|---:|:---:|",
        f"| uniform − FP16 held-out WER | {ci(headroom)} | "
        f"{'PASS' if recoverable_headroom else 'FAIL'} |",
        f"| task − uniform held-out WER | {ci(task_wer)} | "
        f"{'PASS' if task_wer['ci95_upper'] < 0 else 'FAIL'} |",
        f"| task − uniform FP16-trajectory edit | {ci(task_trajectory)} | "
        f"{'PASS' if task_trajectory['ci95_upper'] < 0 else 'FAIL'} |",
        "",
        f"- Uniform stability: `{'PASS' if baseline_feasible else 'FAIL'}`.",
        f"- Task stability: `{'PASS' if decision['task_feasible'] else 'FAIL'}`.",
        f"- Recoverable headroom: `{'PASS' if recoverable_headroom else 'FAIL'}`.",
        f"- Task identifiability: `{'PASS' if task_identifiable else 'FAIL'}`.",
        f"- **Select task metric: `{'YES' if select_task else 'NO'}`.**",
        "",
        "This is deliberately a low-recall safe selector. Failure means abstain; "
        "it does not prove the task metric is harmful.",
    ]
    (args.output_dir / "TRAJECTORY_CANDIDATE_SELECTOR.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
