#!/usr/bin/env python3
"""Unicode-normalized crossed calibration-draw x utterance WER bootstrap."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from rapidfuzz.distance import Levenshtein

from paired_external_bootstrap import normalize, read_rows, summarize


def load_pair(uniform: Path, candidate: Path, dataset: str,
              cap_errors_at_reference: bool = False):
    left = read_rows(uniform / f"{dataset}.jsonl")
    right = read_rows(candidate / f"{dataset}.jsonl")
    lengths, errors_a, errors_b = [], [], []
    for a, b in zip(left, right, strict=True):
        if a["example_id"] != b["example_id"] or a["reference"] != b["reference"]:
            raise ValueError(f"Alignment mismatch for {dataset}/{a['example_id']}")
        reference = normalize(a["reference"]).split()
        lengths.append(len(reference))
        error_a = Levenshtein.distance(reference, normalize(a["prediction"]).split())
        error_b = Levenshtein.distance(reference, normalize(b["prediction"]).split())
        if cap_errors_at_reference:
            error_a = min(error_a, len(reference))
            error_b = min(error_b, len(reference))
        errors_a.append(error_a)
        errors_b.append(error_b)
    return tuple(np.asarray(values, dtype=np.int64) for values in (lengths, errors_a, errors_b))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair", nargs=2, action="append", required=True,
                        metavar=("UNIFORM", "TASK_DENSITY"))
    parser.add_argument("--dataset", action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reps", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument("--baseline-label", default="Uniform GPTQ")
    parser.add_argument("--candidate-label", default="Task density")
    parser.add_argument("--cap-errors-at-reference", action="store_true",
                        help="Diagnostic: cap each utterance edit count at reference length")
    args = parser.parse_args()
    if len(args.pair) < 2:
        raise ValueError("Need at least two calibration draws")
    pairs = [(Path(a), Path(b)) for a, b in args.pair]
    data = {dataset: [load_pair(a, b, dataset, args.cap_errors_at_reference)
                      for a, b in pairs]
            for dataset in args.dataset}
    observed, boot = {}, {}
    draw_macro = np.zeros(len(pairs))
    rng = np.random.default_rng(args.seed)
    # A calibration draw defines one quantized model evaluated on every target
    # dataset.  Resample that draw jointly across datasets; resampling it
    # independently per language would destroy the paired model-level cluster
    # and give an invalid macro interval.  Utterances remain independently
    # resampled within each dataset below.
    shared_draw_indices = rng.integers(
        0, len(pairs), size=(args.reps, len(pairs)), dtype=np.int32
    )
    for dataset, draws in data.items():
        base_lengths = draws[0][0]
        for lengths, _, _ in draws[1:]:
            if not np.array_equal(base_lengths, lengths):
                raise ValueError(f"Reference lengths differ across draws: {dataset}")
        uniform_wers = np.asarray([a.sum() / lengths.sum() for lengths, a, _ in draws])
        task_wers = np.asarray([b.sum() / lengths.sum() for lengths, _, b in draws])
        deltas = task_wers - uniform_wers
        observed[dataset] = {"uniform_wer_draw_mean": float(uniform_wers.mean()),
                             "task_density_wer_draw_mean": float(task_wers.mean()),
                             "delta_draw_mean": float(deltas.mean()),
                             "delta_by_draw": deltas.tolist()}
        draw_macro += deltas / len(args.dataset)
        samples = np.empty(args.reps)
        for start in range(0, args.reps, 128):
            stop = min(args.reps, start + 128)
            size = stop - start
            utterance = rng.integers(0, len(base_lengths),
                                     size=(size, len(base_lengths)), dtype=np.int32)
            denominator = base_lengths[utterance].sum(axis=1)
            draw_samples = np.empty((size, len(draws)))
            for index, (_, errors_a, errors_b) in enumerate(draws):
                draw_samples[:, index] = ((errors_b[utterance].sum(axis=1) -
                                            errors_a[utterance].sum(axis=1)) / denominator)
            draw_indices = shared_draw_indices[start:stop]
            samples[start:stop] = np.take_along_axis(draw_samples, draw_indices, axis=1).mean(axis=1)
        boot[dataset] = samples
    macro_uniform = np.mean([row["uniform_wer_draw_mean"] for row in observed.values()])
    macro_task = np.mean([row["task_density_wer_draw_mean"] for row in observed.values()])
    macro_estimate = macro_task - macro_uniform
    macro_samples = np.mean(np.stack([boot[d] for d in args.dataset]), axis=0)
    payload = {"inference": ("unicode_crossed_calibration_draw_by_utterance_bootstrap"
                              + ("_capped_at_reference" if args.cap_errors_at_reference else "")),
               "error_cap": "reference_length" if args.cap_errors_at_reference else None,
               "baseline_label": args.baseline_label,
               "candidate_label": args.candidate_label,
               "draws": len(pairs), "datasets": args.dataset, "observed": observed,
               "bootstrap": {d: summarize(boot[d], observed[d]["delta_draw_mean"])
                             for d in args.dataset},
               "macro": {"uniform_wer_draw_mean": float(macro_uniform),
                         "task_density_wer_draw_mean": float(macro_task),
                         **summarize(macro_samples, macro_estimate)},
               "macro_delta_by_draw": draw_macro.tolist(),
               "reps": args.reps, "seed": args.seed}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "crossed_external_bootstrap.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n")
    label = "capped-WER diagnostic" if args.cap_errors_at_reference else "WER"
    lines = ["# Unicode-normalized crossed external bootstrap", "",
             f"{len(pairs)} calibration draws; {args.candidate_label} minus "
             f"{args.baseline_label}; {label}.", "",
             f"| Scope | {args.baseline_label} % | {args.candidate_label} % | "
             "Δ pp [95% CI] | p |",
             "|---|---:|---:|---:|---:|"]
    for dataset in args.dataset:
        row, stat = observed[dataset], payload["bootstrap"][dataset]
        lines.append(f"| {dataset} | {100*row['uniform_wer_draw_mean']:.3f} | "
                     f"{100*row['task_density_wer_draw_mean']:.3f} | "
                     f"{100*stat['estimate']:+.3f} [{100*stat['ci95_lower']:+.3f}, "
                     f"{100*stat['ci95_upper']:+.3f}] | {stat['p_two_sided_centered']:.4f} |")
    stat = payload["macro"]
    lines.append(f"| **macro** | **{100*stat['uniform_wer_draw_mean']:.3f}** | "
                 f"**{100*stat['task_density_wer_draw_mean']:.3f}** | "
                 f"**{100*stat['estimate']:+.3f} [{100*stat['ci95_lower']:+.3f}, "
                 f"{100*stat['ci95_upper']:+.3f}]** | **{stat['p_two_sided_centered']:.4f}** |")
    lines.extend(["", "Macro effects by draw (pp): `" +
                  "/".join(f"{100*x:+.3f}" for x in draw_macro) + "`."])
    (args.output_dir / "CROSSED_EXTERNAL_REPORT.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
