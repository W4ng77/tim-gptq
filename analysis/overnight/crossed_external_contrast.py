#!/usr/bin/env python3
"""Named external ASR contrast with shared draw x utterance bootstrap."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from crossed_external_bootstrap import load_pair
from paired_external_bootstrap import summarize


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair", nargs=2, action="append", required=True,
                        metavar=("BASELINE", "CANDIDATE"))
    parser.add_argument("--baseline-label", required=True)
    parser.add_argument("--candidate-label", required=True)
    parser.add_argument("--dataset", action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reps", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--cap-errors-at-reference", action="store_true")
    args = parser.parse_args()
    pairs = [(Path(a), Path(b)) for a, b in args.pair]
    data = {
        dataset: [
            load_pair(a, b, dataset, args.cap_errors_at_reference)
            for a, b in pairs
        ]
        for dataset in args.dataset
    }
    observed, boot = {}, {}
    macro_by_draw = np.zeros(len(pairs), dtype=np.float64)
    rng = np.random.default_rng(args.seed)
    shared_draw_indices = rng.integers(
        0, len(pairs), size=(args.reps, len(pairs)), dtype=np.int32
    )
    for dataset, draws in data.items():
        base_lengths = draws[0][0]
        for lengths, _, _ in draws[1:]:
            if not np.array_equal(base_lengths, lengths):
                raise ValueError(f"Reference lengths differ across draws: {dataset}")
        baseline = np.asarray([
            errors.sum() / lengths.sum() for lengths, errors, _ in draws
        ])
        candidate = np.asarray([
            errors.sum() / lengths.sum() for lengths, _, errors in draws
        ])
        deltas = candidate - baseline
        observed[dataset] = {
            "baseline_wer_draw_mean": float(baseline.mean()),
            "candidate_wer_draw_mean": float(candidate.mean()),
            "delta_draw_mean": float(deltas.mean()),
            "delta_by_draw": deltas.tolist(),
        }
        macro_by_draw += deltas / len(args.dataset)
        samples = np.empty(args.reps, dtype=np.float64)
        for start in range(0, args.reps, 128):
            stop = min(args.reps, start + 128)
            size = stop - start
            utterance = rng.integers(
                0, len(base_lengths), size=(size, len(base_lengths)), dtype=np.int32
            )
            denominator = base_lengths[utterance].sum(axis=1)
            draw_samples = np.empty((size, len(draws)), dtype=np.float64)
            for index, (_, errors_a, errors_b) in enumerate(draws):
                draw_samples[:, index] = (
                    errors_b[utterance].sum(axis=1) - errors_a[utterance].sum(axis=1)
                ) / denominator
            samples[start:stop] = np.take_along_axis(
                draw_samples, shared_draw_indices[start:stop], axis=1
            ).mean(axis=1)
        boot[dataset] = samples

    macro_baseline = float(np.mean([
        row["baseline_wer_draw_mean"] for row in observed.values()
    ]))
    macro_candidate = float(np.mean([
        row["candidate_wer_draw_mean"] for row in observed.values()
    ]))
    macro_estimate = macro_candidate - macro_baseline
    macro_samples = np.mean(np.stack([boot[d] for d in args.dataset]), axis=0)
    payload = {
        "inference": "shared_calibration_draw_by_utterance_bootstrap",
        "contrast": (
            f"{args.candidate_label}_minus_{args.baseline_label}"
            + ("_capped_at_reference" if args.cap_errors_at_reference else "")
        ),
        "error_cap": "reference_length" if args.cap_errors_at_reference else None,
        "baseline_label": args.baseline_label,
        "candidate_label": args.candidate_label,
        "draws": len(pairs),
        "datasets": args.dataset,
        "observed": observed,
        "bootstrap": {
            dataset: summarize(boot[dataset], observed[dataset]["delta_draw_mean"])
            for dataset in args.dataset
        },
        "macro": {
            "baseline_wer_draw_mean": macro_baseline,
            "candidate_wer_draw_mean": macro_candidate,
            **summarize(macro_samples, macro_estimate),
        },
        "macro_delta_by_draw": macro_by_draw.tolist(),
        "reps": args.reps,
        "seed": args.seed,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "crossed_external_contrast.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = [
        "# Named crossed external contrast", "",
        f"Candidate minus baseline: **{args.candidate_label} − {args.baseline_label}**; "
        "positive WER difference favors the baseline.", "",
        ("Per-utterance edit counts are capped at reference length." if
         args.cap_errors_at_reference else "Raw corpus WER."), "",
        f"{len(pairs)} calibration draw(s); model-level draw resampled jointly across datasets.", "",
        "| Scope | Baseline % | Candidate % | Δ pp [95% CI] | p |",
        "|---|---:|---:|---:|---:|",
    ]
    for dataset in args.dataset:
        row = observed[dataset]
        stat = payload["bootstrap"][dataset]
        lines.append(
            f"| {dataset} | {100*row['baseline_wer_draw_mean']:.3f} | "
            f"{100*row['candidate_wer_draw_mean']:.3f} | {100*stat['estimate']:+.3f} "
            f"[{100*stat['ci95_lower']:+.3f}, {100*stat['ci95_upper']:+.3f}] | "
            f"{stat['p_two_sided_centered']:.4f} |"
        )
    stat = payload["macro"]
    lines.append(
        f"| **macro** | **{100*stat['baseline_wer_draw_mean']:.3f}** | "
        f"**{100*stat['candidate_wer_draw_mean']:.3f}** | "
        f"**{100*stat['estimate']:+.3f} [{100*stat['ci95_lower']:+.3f}, "
        f"{100*stat['ci95_upper']:+.3f}]** | **{stat['p_two_sided_centered']:.4f}** |"
    )
    lines.extend([
        "", "Effects by calibration draw (pp): `" +
        "/".join(f"{100*x:+.3f}" for x in macro_by_draw) + "`.",
    ])
    (args.output_dir / "CROSSED_EXTERNAL_CONTRAST.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
