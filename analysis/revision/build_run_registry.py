#!/usr/bin/env python3
"""Build a deterministic inventory of every status/config-tracked run.

The registry is descriptive only.  It never decides which runs enter an
analysis.  Statistical scripts must continue to require
``status.json.state == \"completed\"`` and their pre-registered protocol.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
from pathlib import Path
from statistics import fmean


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def compact(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple)):
        return ",".join(str(item) for item in value)
    return str(value)


def discover(runs: Path) -> list[Path]:
    markers = ("status.json", "config.json", "metrics.json")
    return sorted({path.parent for name in markers for path in runs.rglob(name)})


def build_row(runs: Path, run_dir: Path) -> dict[str, str]:
    status_path = run_dir / "status.json"
    config_path = run_dir / "config.json"
    metrics_path = run_dir / "metrics.json"
    allocation_path = run_dir / "allocation.json"
    status = read_json(status_path)
    config = read_json(config_path)
    metrics = read_json(metrics_path)
    allocation = read_json(allocation_path)
    evaluations = metrics.get("evaluations", [])
    if not isinstance(evaluations, list):
        evaluations = []
    valid_evals = [
        item
        for item in evaluations
        if isinstance(item, dict) and isinstance(item.get("wer"), (int, float))
    ]
    macro_wer = fmean(item["wer"] for item in valid_evals) * 100 if valid_evals else None
    coverage = ",".join(
        f"{item.get('dataset', '?')}:{item.get('num_examples', '?')}"
        for item in valid_evals
    )
    relative = run_dir.relative_to(runs)
    parts = relative.parts
    marker_mtimes = [
        path.stat().st_mtime
        for path in (status_path, config_path, metrics_path, allocation_path)
        if path.exists()
    ]
    artifact_time = ""
    if marker_mtimes:
        artifact_time = dt.datetime.fromtimestamp(
            max(marker_mtimes), tz=dt.timezone.utc
        ).isoformat().replace("+00:00", "Z")
    state = compact(status.get("state")) or "missing-status"
    return {
        "namespace": parts[0] if parts else "",
        "run_name": compact(config.get("run_name")) or run_dir.name,
        "state": state,
        "model": compact(config.get("model_alias") or config.get("model")),
        "method": compact(config.get("method") or config.get("mode")),
        "wbits": compact(config.get("wbits")),
        "target_average_bits": compact(config.get("target_average_bits")),
        "actual_average_bits": compact(
            allocation.get("actual_parameter_weighted_average_bits")
        ),
        "reference_bits": compact(config.get("reference_bits")),
        "groupsize": compact(config.get("groupsize") or config.get("group_size")),
        "nsamples": compact(config.get("nsamples")),
        "quant_scope": compact(config.get("quant_scope")),
        "frame_weighting": compact(config.get("frame_weighting")),
        "sequence_support": compact(config.get("sequence_support")),
        "sequence_prompt_template": compact(config.get("sequence_prompt_template")),
        "sequence_support_seed": compact(config.get("sequence_support_seed")),
        "interface_bridge": compact(config.get("interface_bridge")),
        "rotate": compact(config.get("rotate")),
        "clip_max": compact(config.get("propagated_clip_max")),
        "calibration_seed": compact(config.get("seed")),
        "quantization_seed": compact(config.get("quantization_seed")),
        "permutation": compact(config.get("task_weight_permutation")),
        "permutation_seed": compact(config.get("task_weight_permutation_seed")),
        "datasets": compact(config.get("datasets")),
        "manifest_dir": compact(config.get("eval_manifest_dir")),
        "eval_samples": compact(config.get("eval_samples")),
        "official_source_commit": compact(
            (
                (config.get("official_source") or {}).get("commit")
                if isinstance(config.get("official_source"), dict)
                else None
            )
            or config.get("official_omniquant_commit")
        ),
        "eval_coverage": coverage,
        "macro_wer_percent": "" if macro_wer is None else f"{macro_wer:.6f}",
        "quantization_seconds": compact(metrics.get("quantization_seconds")),
        "total_seconds": compact(metrics.get("total_seconds")),
        "artifact_updated_utc": artifact_time,
        "run_dir": str(relative),
        "has_config": compact(config_path.exists()),
        "has_metrics": compact(metrics_path.exists()),
        "has_allocation": compact(allocation_path.exists()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, default=Path("runs"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("analysis/revision_20260801/RUN_REGISTRY.tsv"),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("analysis/revision_20260801/RUN_REGISTRY_SUMMARY.json"),
    )
    args = parser.parse_args()

    rows = [build_row(args.runs, run_dir) for run_dir in discover(args.runs)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, dialect="excel-tab")
        writer.writeheader()
        writer.writerows(rows)

    by_state: dict[str, int] = {}
    by_namespace: dict[str, int] = {}
    for row in rows:
        by_state[row["state"]] = by_state.get(row["state"], 0) + 1
        by_namespace[row["namespace"]] = by_namespace.get(row["namespace"], 0) + 1
    summary = {
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
        "runs_root": str(args.runs.resolve()),
        "row_count": len(rows),
        "by_state": dict(sorted(by_state.items())),
        "by_namespace": dict(sorted(by_namespace.items())),
        "policy": "Inventory only; analyses require terminal status and pre-registered inclusion rules.",
    }
    args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
