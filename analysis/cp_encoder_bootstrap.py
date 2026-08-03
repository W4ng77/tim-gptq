#!/usr/bin/env python3
"""Audit and bootstrap the W2J consumer/propagation encoder factorial.

The experiment has four cells per model:

    UB: uniform terminal metric, broadcast to the encoder
    AB: attention terminal metric, broadcast to the encoder
    UV: uniform terminal metric, propagated by VJP
    AV: attention terminal metric, propagated by VJP

This script first performs strict provenance and pairing checks. It then
computes word-level edit counts per utterance and runs a dataset-stratified,
utterance-level paired bootstrap. Every bootstrap draw is shared by all four
cells (and both model sizes), so the factorial interaction

    AV - AB - UV + UB

is computed from genuinely paired four-cell resamples.

The reported macro WER is the unweighted mean of the five corpus WERs, matching
the workbench convention. The bootstrap is conditional on the fixed evaluated
utterances and the single quantization/calibration seed represented by these
runs; it does not measure between-seed variation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from rapidfuzz.distance import Levenshtein


ARMS = {
    "UB": ("uniform-broadcast", "none"),
    "AB": ("attention-broadcast", "attention"),
    "UV": ("uniform-vjp", "propagated-uniform"),
    "AV": ("attention-vjp", "propagated"),
}
ARM_ORDER = ("UB", "AB", "UV", "AV")
CONTRASTS = {
    "AB-UB": {"AB": 1.0, "UB": -1.0},
    "UV-UB": {"UV": 1.0, "UB": -1.0},
    "AV-UB": {"AV": 1.0, "UB": -1.0},
    "AV-AB-UV+UB": {"AV": 1.0, "AB": -1.0, "UV": -1.0, "UB": 1.0},
}
PRIMARY_CONTRASTS = ("AB-UB", "UV-UB", "AV-UB")
DATASETS = (
    "librispeech-clean",
    "librispeech-other",
    "spgispeech",
    "voxpopuli",
    "gigaspeech",
)
SOURCE_KEYS = (
    "git_commit",
    "source_tree_sha256",
    "python",
    "torch",
    "transformers",
    "datasets",
)
ALLOWED_CONFIG_DIFFERENCES_WITHIN_MODEL = {"frame_weighting", "run_name"}
ALLOWED_CONFIG_DIFFERENCES_ACROSS_MODELS = {
    "frame_weighting",
    "run_name",
    "model",
    "model_alias",
}


@dataclass(frozen=True)
class RunRecord:
    model: str
    arm: str
    path: Path
    config: dict[str, Any]
    environment: dict[str, Any]
    quantization: dict[str, Any]
    calibration: dict[str, Any]
    metrics: dict[str, Any]


@dataclass
class DatasetRows:
    ids: dict[str, list[str]]
    references: dict[str, list[str]]
    predictions: dict[str, dict[str, dict[str, list[str]]]]


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    workbench = script_dir.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=workbench / "runs" / "w2j-cp-encoder",
        help="Directory containing the eight completed W2J run directories.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=script_dir / "cp_encoder",
        help="Directory for the audit, CSVs, and Markdown report.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["whisper-tiny", "whisper-base"],
        help="Model aliases to analyze; each must have all four factorial cells.",
    )
    parser.add_argument(
        "--run-name-suffix",
        default="",
        help=(
            "Optional suffix appended after each canonical arm name, e.g. "
            "'-calseed20260730' for held-out calibration-draw runs."
        ),
    )
    parser.add_argument(
        "--bootstrap-reps",
        type=int,
        default=10_000,
        help="Number of paired stratified bootstrap replicates.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260731,
        help="NumPy bootstrap RNG seed.",
    )
    parser.add_argument(
        "--bootstrap-batch-size",
        type=int,
        default=256,
        help="Replicates generated per batch to bound memory use.",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}")
            rows.append(row)
    return rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def without_keys(value: dict[str, Any], keys: set[str]) -> dict[str, Any]:
    return {key: item for key, item in value.items() if key not in keys}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def expected_run_path(
    runs_root: Path, model: str, arm: str, run_name_suffix: str = ""
) -> Path:
    suffix, _ = ARMS[arm]
    return runs_root / f"{model}-gptq-encoder-w3-{suffix}{run_name_suffix}"


def discover_and_audit_runs(
    runs_root: Path, models: list[str], run_name_suffix: str = ""
) -> tuple[dict[str, dict[str, RunRecord]], Path, list[str]]:
    validations: list[str] = []
    runs: dict[str, dict[str, RunRecord]] = {}
    required_files = (
        "status.json",
        "config.json",
        "environment.json",
        "quantization.json",
        "calibration.json",
        "metrics.json",
    )

    for model in models:
        runs[model] = {}
        for arm in ARM_ORDER:
            path = expected_run_path(runs_root, model, arm, run_name_suffix)
            require(path.is_dir(), f"Missing expected run directory: {path}")
            for filename in required_files:
                require((path / filename).is_file(), f"Missing {path / filename}")
            for dataset in DATASETS:
                require(
                    (path / f"{dataset}.jsonl").is_file(),
                    f"Missing {path / f'{dataset}.jsonl'}",
                )

            status = read_json(path / "status.json")
            status_value = status.get("status", status.get("state"))
            require(
                status_value == "completed",
                f"{path.name} status is {status_value!r}, not 'completed'",
            )
            config = read_json(path / "config.json")
            environment = read_json(path / "environment.json")
            quantization = read_json(path / "quantization.json")
            calibration = read_json(path / "calibration.json")
            metrics = read_json(path / "metrics.json")
            _, expected_frame_mode = ARMS[arm]

            require(
                config.get("run_name") == path.name,
                f"{path.name}: config run_name does not match directory",
            )
            require(
                config.get("model_alias") == model,
                f"{path.name}: model_alias is not {model!r}",
            )
            require(config.get("mode") == "gptq", f"{path.name}: mode is not gptq")
            require(
                config.get("method") == "gptq", f"{path.name}: method is not gptq"
            )
            require(
                config.get("quant_scope") == "encoder",
                f"{path.name}: quant_scope is not encoder",
            )
            require(config.get("wbits") == 3, f"{path.name}: wbits is not 3")
            require(
                config.get("frame_weighting") == expected_frame_mode,
                f"{path.name}: frame_weighting should be {expected_frame_mode!r}",
            )
            require(config.get("eval") is True, f"{path.name}: eval is not true")
            require(
                tuple(config.get("datasets", [])) == DATASETS,
                f"{path.name}: dataset list/order does not match expected five domains",
            )

            allocation = quantization.get("bit_allocation", {})
            require(
                allocation.get("effective_bits_over_quantized_weights") == 3,
                f"{path.name}: effective quantized weight bits is not 3",
            )
            by_bits = allocation.get("quantized_weight_numel_by_bits", {})
            require(
                set(map(str, by_bits)) == {"3"},
                f"{path.name}: bit allocation contains non-W3 weights: {by_bits}",
            )
            require(
                int(by_bits["3"]) == int(allocation.get("quantized_weight_numel", -1)),
                f"{path.name}: W3 parameter count does not equal total quantized count",
            )
            require(
                calibration.get("num_examples") == config.get("nsamples"),
                f"{path.name}: calibration count does not match nsamples",
            )
            require(
                len(calibration.get("example_ids", [])) == config.get("nsamples"),
                f"{path.name}: calibration ID count does not match nsamples",
            )

            runs[model][arm] = RunRecord(
                model=model,
                arm=arm,
                path=path,
                config=config,
                environment=environment,
                quantization=quantization,
                calibration=calibration,
                metrics=metrics,
            )

    validations.append(
        f"All {len(models) * len(ARM_ORDER)} expected runs exist and have completed status."
    )

    all_records = [
        runs[model][arm] for model in models for arm in ARM_ORDER
    ]
    source_reference = {
        key: all_records[0].environment.get(key) for key in SOURCE_KEYS
    }
    for record in all_records[1:]:
        actual = {key: record.environment.get(key) for key in SOURCE_KEYS}
        require(
            actual == source_reference,
            f"{record.path.name}: source/software provenance differs: "
            f"{actual!r} != {source_reference!r}",
        )
    require(
        source_reference["git_commit"] is not None
        and source_reference["source_tree_sha256"] is not None,
        "Source provenance is missing git_commit or source_tree_sha256",
    )
    validations.append(
        "Source and software provenance keys match across all eight runs "
        f"(commit {source_reference['git_commit']}, tree "
        f"{source_reference['source_tree_sha256']})."
    )

    for model in models:
        config_reference = without_keys(
            runs[model]["UB"].config, ALLOWED_CONFIG_DIFFERENCES_WITHIN_MODEL
        )
        allocation_reference = runs[model]["UB"].quantization.get(
            "bit_allocation", {}
        )
        for arm in ARM_ORDER:
            record = runs[model][arm]
            require(
                without_keys(
                    record.config, ALLOWED_CONFIG_DIFFERENCES_WITHIN_MODEL
                )
                == config_reference,
                f"{record.path.name}: config differs beyond frame_weighting/run_name",
            )
            require(
                record.quantization.get("bit_allocation", {})
                == allocation_reference,
                f"{record.path.name}: bit budget differs within {model}",
            )
        validations.append(
            f"{model}: configs match except intended arm fields and all four bit "
            "allocation records are identical."
        )

    global_config_reference = without_keys(
        runs[models[0]]["UB"].config, ALLOWED_CONFIG_DIFFERENCES_ACROSS_MODELS
    )
    for model in models:
        for arm in ARM_ORDER:
            record = runs[model][arm]
            require(
                without_keys(
                    record.config, ALLOWED_CONFIG_DIFFERENCES_ACROSS_MODELS
                )
                == global_config_reference,
                f"{record.path.name}: cross-model config differs beyond model/arm fields",
            )
    validations.append(
        "Cross-model configs match after removing only model identity and intended arm fields."
    )

    calibration_ids = all_records[0].calibration["example_ids"]
    for record in all_records[1:]:
        require(
            record.calibration["example_ids"] == calibration_ids,
            f"{record.path.name}: calibration IDs/order differ",
        )
    validations.append(
        f"All runs use the same {len(calibration_ids)} calibration IDs in the same order."
    )

    manifest_roots = {
        Path(record.config["eval_manifest_dir"]).expanduser().resolve()
        for record in all_records
    }
    require(
        len(manifest_roots) == 1,
        f"Runs use multiple evaluation manifest roots: {manifest_roots}",
    )
    manifest_root = next(iter(manifest_roots))
    require(manifest_root.is_dir(), f"Manifest directory does not exist: {manifest_root}")

    return runs, manifest_root, validations


def load_and_audit_evaluations(
    runs: dict[str, dict[str, RunRecord]],
    manifest_root: Path,
    validations: list[str],
) -> DatasetRows:
    models = list(runs)
    canonical_ids: dict[str, list[str]] = {}
    canonical_references: dict[str, list[str]] = {}
    predictions: dict[str, dict[str, dict[str, list[str]]]] = {
        model: {arm: {} for arm in ARM_ORDER} for model in models
    }

    for dataset in DATASETS:
        manifest_path = manifest_root / f"{dataset}.json"
        require(manifest_path.is_file(), f"Missing evaluation manifest: {manifest_path}")
        manifest = read_json(manifest_path)
        require(
            manifest.get("dataset_alias") == dataset,
            f"{manifest_path}: dataset_alias mismatch",
        )
        selected = manifest.get("selected")
        require(isinstance(selected, list), f"{manifest_path}: selected is not a list")
        manifest_indices = [int(item["index"]) for item in selected]
        manifest_ids = [str(item["id"]) for item in selected]
        require(
            len(selected) == int(manifest.get("num_selected", -1)),
            f"{manifest_path}: selected length does not match num_selected",
        )
        require(
            len(set(manifest_ids)) == len(manifest_ids),
            f"{manifest_path}: duplicate selected IDs",
        )
        require(
            len(set(manifest_indices)) == len(manifest_indices),
            f"{manifest_path}: duplicate selected source indices",
        )

        first_ids: list[str] | None = None
        first_references: list[str] | None = None
        for model in models:
            for arm in ARM_ORDER:
                record = runs[model][arm]
                rows = read_jsonl(record.path / f"{dataset}.jsonl")
                ids = [str(row.get("example_id")) for row in rows]
                indices = [int(row.get("index", -1)) for row in rows]
                references = [str(row.get("reference")) for row in rows]
                arm_predictions = [str(row.get("prediction")) for row in rows]

                require(
                    indices == list(range(len(rows))),
                    f"{record.path.name}/{dataset}: JSONL output indices are not "
                    "contiguous evaluation positions",
                )
                require(
                    ids == manifest_ids,
                    f"{record.path.name}/{dataset}: JSONL IDs/order do not match manifest",
                )
                require(
                    all("reference" in row and "prediction" in row for row in rows),
                    f"{record.path.name}/{dataset}: missing reference or prediction",
                )
                if first_ids is None:
                    first_ids = ids
                    first_references = references
                else:
                    require(
                        ids == first_ids,
                        f"{record.path.name}/{dataset}: IDs differ across cells",
                    )
                    require(
                        references == first_references,
                        f"{record.path.name}/{dataset}: references differ across cells",
                    )
                predictions[model][arm][dataset] = arm_predictions

                metric_entries = {
                    item["dataset"]: item
                    for item in record.metrics.get("evaluations", [])
                }
                require(
                    set(metric_entries) == set(DATASETS),
                    f"{record.path.name}: metrics dataset set mismatch",
                )
                metric = metric_entries[dataset]
                require(
                    int(metric.get("num_examples", -1)) == len(rows),
                    f"{record.path.name}/{dataset}: metrics num_examples mismatch",
                )
                require(
                    Path(metric["manifest"]).expanduser().resolve()
                    == manifest_path.resolve(),
                    f"{record.path.name}/{dataset}: metrics manifest path mismatch",
                )

        assert first_ids is not None and first_references is not None
        canonical_ids[dataset] = first_ids
        canonical_references[dataset] = first_references
        validations.append(
            f"{dataset}: {len(first_ids)} manifest IDs and references align exactly "
            "across all eight cells; JSONL evaluation positions are contiguous."
        )

    return DatasetRows(
        ids=canonical_ids,
        references=canonical_references,
        predictions=predictions,
    )


def compute_error_arrays(
    runs: dict[str, dict[str, RunRecord]],
    rows: DatasetRows,
) -> tuple[
    dict[str, np.ndarray],
    dict[str, dict[str, dict[str, np.ndarray]]],
    list[dict[str, Any]],
]:
    references_by_dataset = rows.references
    predictions = rows.predictions
    ref_lengths: dict[str, np.ndarray] = {}
    errors: dict[str, dict[str, dict[str, np.ndarray]]] = {
        model: {arm: {} for arm in ARM_ORDER} for model in runs
    }
    wer_rows: list[dict[str, Any]] = []

    for dataset in DATASETS:
        references = references_by_dataset[dataset]
        tokenized_references = [reference.split() for reference in references]
        lengths = np.asarray(
            [len(tokens) for tokens in tokenized_references], dtype=np.int64
        )
        require(
            int(lengths.sum()) > 0,
            f"{dataset}: total normalized reference word count is zero",
        )
        ref_lengths[dataset] = lengths

        for model in runs:
            for arm in ARM_ORDER:
                arm_predictions = predictions[model][arm][dataset]
                require(
                    len(arm_predictions) == len(references),
                    f"{model}/{arm}/{dataset}: prediction count mismatch",
                )
                arm_errors = np.fromiter(
                    (
                        Levenshtein.distance(reference_tokens, prediction.split())
                        for reference_tokens, prediction in zip(
                            tokenized_references, arm_predictions, strict=True
                        )
                    ),
                    dtype=np.int64,
                    count=len(references),
                )
                errors[model][arm][dataset] = arm_errors
                error_total = int(arm_errors.sum())
                ref_total = int(lengths.sum())
                observed_wer = error_total / ref_total
                metric = next(
                    item
                    for item in runs[model][arm].metrics["evaluations"]
                    if item["dataset"] == dataset
                )
                require(
                    math.isclose(
                        observed_wer,
                        float(metric["wer"]),
                        rel_tol=0.0,
                        abs_tol=5e-15,
                    ),
                    f"{model}/{arm}/{dataset}: recomputed WER {observed_wer} "
                    f"does not match metrics.json {metric['wer']}",
                )
                wer_rows.append(
                    {
                        "model": model,
                        "arm": arm,
                        "scope": "dataset",
                        "dataset": dataset,
                        "num_examples": len(references),
                        "reference_words": ref_total,
                        "word_errors": error_total,
                        "wer": observed_wer,
                        "wer_percent": 100.0 * observed_wer,
                    }
                )

    for model in runs:
        for arm in ARM_ORDER:
            domain_rows = [
                row
                for row in wer_rows
                if row["model"] == model and row["arm"] == arm
            ]
            macro = float(np.mean([row["wer"] for row in domain_rows]))
            wer_rows.append(
                {
                    "model": model,
                    "arm": arm,
                    "scope": "macro",
                    "dataset": "macro",
                    "num_examples": sum(row["num_examples"] for row in domain_rows),
                    "reference_words": "",
                    "word_errors": "",
                    "wer": macro,
                    "wer_percent": 100.0 * macro,
                }
            )

    return ref_lengths, errors, wer_rows


def run_bootstrap(
    ref_lengths: dict[str, np.ndarray],
    errors: dict[str, dict[str, dict[str, np.ndarray]]],
    reps: int,
    seed: int,
    batch_size: int,
) -> dict[str, dict[str, dict[str, np.ndarray]]]:
    require(reps >= 100, "--bootstrap-reps must be at least 100")
    require(batch_size >= 1, "--bootstrap-batch-size must be positive")
    rng = np.random.default_rng(seed)
    boot: dict[str, dict[str, dict[str, np.ndarray]]] = {
        model: {
            arm: {
                dataset: np.empty(reps, dtype=np.float64) for dataset in DATASETS
            }
            for arm in ARM_ORDER
        }
        for model in errors
    }

    for dataset in DATASETS:
        lengths = ref_lengths[dataset]
        n = len(lengths)
        for start in range(0, reps, batch_size):
            stop = min(start + batch_size, reps)
            draw = rng.integers(
                0, n, size=(stop - start, n), dtype=np.int32, endpoint=False
            )
            denominator = lengths[draw].sum(axis=1, dtype=np.int64)
            require(
                bool(np.all(denominator > 0)),
                f"{dataset}: zero bootstrap reference denominator",
            )
            for model in errors:
                for arm in ARM_ORDER:
                    numerator = errors[model][arm][dataset][draw].sum(
                        axis=1, dtype=np.int64
                    )
                    boot[model][arm][dataset][start:stop] = (
                        numerator / denominator
                    )
    return boot


def linear_combination(
    values: dict[str, float] | dict[str, np.ndarray], coefficients: dict[str, float]
) -> float | np.ndarray:
    result: float | np.ndarray = 0.0
    for arm, coefficient in coefficients.items():
        result = result + coefficient * values[arm]
    return result


def centered_bootstrap_p(
    samples: np.ndarray, estimate: float
) -> float:
    """Two-sided, plus-one bootstrap p-value under a centered null."""
    deviations = np.abs(samples - estimate)
    exceedances = int(np.count_nonzero(deviations >= abs(estimate)))
    return (exceedances + 1.0) / (len(samples) + 1.0)


def holm_adjust(p_values: list[float]) -> list[float]:
    count = len(p_values)
    order = sorted(range(count), key=p_values.__getitem__)
    adjusted = [0.0] * count
    running_max = 0.0
    for rank, index in enumerate(order):
        candidate = min(1.0, (count - rank) * p_values[index])
        running_max = max(running_max, candidate)
        adjusted[index] = running_max
    return adjusted


def make_contrast_rows(
    wer_rows: list[dict[str, Any]],
    boot: dict[str, dict[str, dict[str, np.ndarray]]],
) -> list[dict[str, Any]]:
    observed: dict[str, dict[str, dict[str, float]]] = {}
    for row in wer_rows:
        observed.setdefault(row["model"], {}).setdefault(row["arm"], {})[
            row["dataset"]
        ] = float(row["wer"])

    contrast_rows: list[dict[str, Any]] = []
    for model in boot:
        for dataset in (*DATASETS, "macro"):
            observed_by_arm = {
                arm: observed[model][arm][dataset] for arm in ARM_ORDER
            }
            if dataset == "macro":
                samples_by_arm = {
                    arm: np.mean(
                        np.stack([boot[model][arm][name] for name in DATASETS]),
                        axis=0,
                    )
                    for arm in ARM_ORDER
                }
                scope = "macro"
            else:
                samples_by_arm = {
                    arm: boot[model][arm][dataset] for arm in ARM_ORDER
                }
                scope = "dataset"

            for contrast, coefficients in CONTRASTS.items():
                estimate = float(linear_combination(observed_by_arm, coefficients))
                samples = np.asarray(
                    linear_combination(samples_by_arm, coefficients),
                    dtype=np.float64,
                )
                lower, upper = np.quantile(samples, [0.025, 0.975])
                contrast_rows.append(
                    {
                        "model": model,
                        "scope": scope,
                        "dataset": dataset,
                        "contrast": contrast,
                        "primary": contrast in PRIMARY_CONTRASTS,
                        "estimate": estimate,
                        "ci95_lower": float(lower),
                        "ci95_upper": float(upper),
                        "estimate_pp": 100.0 * estimate,
                        "ci95_lower_pp": 100.0 * float(lower),
                        "ci95_upper_pp": 100.0 * float(upper),
                        "bootstrap_p_two_sided": centered_bootstrap_p(
                            samples, estimate
                        ),
                        "holm_p_within_model_3": "",
                        "holm_p_primary_family_6": "",
                        "holm_p_all_macro_8": "",
                    }
                )

    macro_rows = [row for row in contrast_rows if row["scope"] == "macro"]
    for model in boot:
        family = [
            row
            for row in macro_rows
            if row["model"] == model and row["contrast"] in PRIMARY_CONTRASTS
        ]
        adjusted = holm_adjust(
            [float(row["bootstrap_p_two_sided"]) for row in family]
        )
        for row, value in zip(family, adjusted, strict=True):
            row["holm_p_within_model_3"] = value

    primary_family = [
        row for row in macro_rows if row["contrast"] in PRIMARY_CONTRASTS
    ]
    adjusted_primary = holm_adjust(
        [float(row["bootstrap_p_two_sided"]) for row in primary_family]
    )
    for row, value in zip(primary_family, adjusted_primary, strict=True):
        row["holm_p_primary_family_6"] = value

    adjusted_all = holm_adjust(
        [float(row["bootstrap_p_two_sided"]) for row in macro_rows]
    )
    for row, value in zip(macro_rows, adjusted_all, strict=True):
        row["holm_p_all_macro_8"] = value
    return contrast_rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    require(bool(rows), f"Refusing to write empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def format_p(value: float | str) -> str:
    if value == "":
        return "—"
    number = float(value)
    return "<0.0001" if number < 0.0001 else f"{number:.4f}"


def format_interval(row: dict[str, Any]) -> str:
    return (
        f"{row['estimate_pp']:+.3f} "
        f"[{row['ci95_lower_pp']:+.3f}, {row['ci95_upper_pp']:+.3f}]"
    )


def build_report(
    runs: dict[str, dict[str, RunRecord]],
    validations: list[str],
    wer_rows: list[dict[str, Any]],
    contrast_rows: list[dict[str, Any]],
    reps: int,
    seed: int,
) -> str:
    models = list(runs)
    macro_wer = {
        (row["model"], row["arm"]): row
        for row in wer_rows
        if row["scope"] == "macro"
    }
    macro_contrasts = {
        (row["model"], row["contrast"]): row
        for row in contrast_rows
        if row["scope"] == "macro"
    }
    lines = [
        "# W2J C×P encoder factorial: paired bootstrap",
        "",
        "## Result",
        "",
        f"Analysis uses {reps:,} utterance-level bootstrap replicates (seed "
        f"`{seed}`), stratified by dataset. Within every stratum and replicate, "
        "the identical sampled utterance indices are used for UB, AB, UV, and "
        "AV; the interaction is therefore a paired four-output statistic.",
        "",
        "WER values below are percentages. Contrast values are percentage-point "
        "differences, and negative values favor the named arm over UB. Macro WER "
        "is the unweighted mean of the five dataset corpus WERs.",
        "",
        "### Macro WER",
        "",
        "| Model | UB | AB | UV | AV |",
        "|---|---:|---:|---:|---:|",
    ]
    for model in models:
        values = [
            f"{macro_wer[(model, arm)]['wer_percent']:.3f}" for arm in ARM_ORDER
        ]
        lines.append(f"| {model} | " + " | ".join(values) + " |")

    lines += [
        "",
        "### Primary macro contrasts",
        "",
        "The confirmatory family contains six tests: three vs-UB contrasts for "
        "each of two models. `Holm-6` controls family-wise error across all six; "
        "`Holm-3` is also shown for the pre-specified three contrasts within each "
        "model.",
        "",
        "| Model | Contrast | Δ WER pp [95% CI] | raw p | Holm-3 | Holm-6 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for model in models:
        for contrast in PRIMARY_CONTRASTS:
            row = macro_contrasts[(model, contrast)]
            lines.append(
                f"| {model} | {contrast} | {format_interval(row)} | "
                f"{format_p(row['bootstrap_p_two_sided'])} | "
                f"{format_p(row['holm_p_within_model_3'])} | "
                f"{format_p(row['holm_p_primary_family_6'])} |"
            )

    lines += [
        "",
        "### Factorial interaction",
        "",
        "`AV-AB-UV+UB` tests whether the effect of attention conditioning changes "
        "when switching from broadcast to VJP propagation. A negative interaction "
        "means the combined AV cell is better (lower WER) than the additive main-"
        "effects prediction.",
        "",
        "| Model | Interaction Δ WER pp [95% CI] | raw p | Holm over all 8 macro tests |",
        "|---|---:|---:|---:|",
    ]
    for model in models:
        row = macro_contrasts[(model, "AV-AB-UV+UB")]
        lines.append(
            f"| {model} | {format_interval(row)} | "
            f"{format_p(row['bootstrap_p_two_sided'])} | "
            f"{format_p(row['holm_p_all_macro_8'])} |"
        )

    lines += ["", "## Per-dataset contrasts", ""]
    for model in models:
        lines += [
            f"### {model}",
            "",
            "| Dataset | AB-UB | UV-UB | AV-UB | Interaction |",
            "|---|---:|---:|---:|---:|",
        ]
        for dataset in DATASETS:
            cells = []
            for contrast in CONTRASTS:
                row = next(
                    item
                    for item in contrast_rows
                    if item["model"] == model
                    and item["dataset"] == dataset
                    and item["contrast"] == contrast
                )
                cells.append(format_interval(row))
            lines.append(f"| {dataset} | " + " | ".join(cells) + " |")
        lines += [
            "",
            "Per-dataset intervals are exploratory and are not multiplicity-adjusted.",
            "",
        ]

    source = runs[models[0]]["UB"].environment
    lines += [
        "## Audit",
        "",
    ]
    lines.extend(f"- {validation}" for validation in validations)
    lines += [
        "",
        "Bit budgets:",
        "",
        "| Model | quantized weights | fraction of all params | effective bits/all params |",
        "|---|---:|---:|---:|",
    ]
    for model in models:
        allocation = runs[model]["UB"].quantization["bit_allocation"]
        lines.append(
            f"| {model} | {int(allocation['quantized_weight_numel']):,} @ W3 | "
            f"{100.0 * float(allocation['quantized_parameter_fraction']):.3f}% | "
            f"{float(allocation['effective_bits_over_all_parameters']):.6f} |"
        )
    lines += [
        "",
        f"Common source commit: `{source['git_commit']}`  ",
        f"Common source-tree SHA-256: `{source['source_tree_sha256']}`",
        "",
        "## Statistical definition and limitation",
        "",
        "For each utterance and cell, word errors are the word-token Levenshtein "
        "distance between the stored normalized reference and prediction. A "
        "dataset corpus WER is the sum of sampled word errors divided by the sum "
        "of sampled reference words. The percentile interval uses the 2.5th and "
        "97.5th bootstrap percentiles. Two-sided p-values use a centered bootstrap "
        "null with a plus-one correction; Holm adjustments are applied only to "
        "the stated macro families.",
        "",
        "Inference is conditional on these five fixed evaluation manifests and one "
        "calibration/quantization seed. Utterance bootstrap captures evaluation-"
        "sample uncertainty, not variation from recalibrating or requantizing the "
        "model. Seed-level claims require independent quantization repeats.",
        "",
        "Machine-readable outputs: `run_audit.csv`, `wer_by_arm.csv`, "
        "`bootstrap_contrasts.csv`, and `audit.json`.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    runs_root = args.runs_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    runs, manifest_root, validations = discover_and_audit_runs(
        runs_root, list(args.models), args.run_name_suffix
    )
    rows = load_and_audit_evaluations(runs, manifest_root, validations)
    ref_lengths, errors, wer_rows = compute_error_arrays(runs, rows)
    validations.append(
        "Stored metrics WER matches recomputation from every JSONL exactly "
        "(absolute tolerance 5e-15)."
    )
    boot = run_bootstrap(
        ref_lengths,
        errors,
        reps=args.bootstrap_reps,
        seed=args.seed,
        batch_size=args.bootstrap_batch_size,
    )
    contrast_rows = make_contrast_rows(wer_rows, boot)

    output_dir.mkdir(parents=True, exist_ok=True)
    run_audit_rows: list[dict[str, Any]] = []
    for model in args.models:
        for arm in ARM_ORDER:
            record = runs[model][arm]
            allocation = record.quantization["bit_allocation"]
            run_audit_rows.append(
                {
                    "model": model,
                    "arm": arm,
                    "run_name": record.path.name,
                    "status": "completed",
                    "git_commit": record.environment["git_commit"],
                    "source_tree_sha256": record.environment["source_tree_sha256"],
                    "config_sha256": sha256_file(record.path / "config.json"),
                    "calibration_ids_sha256": sha256_json(
                        record.calibration["example_ids"]
                    ),
                    "wbits": record.config["wbits"],
                    "quant_scope": record.config["quant_scope"],
                    "quantized_weight_numel": allocation["quantized_weight_numel"],
                    "quantized_parameter_fraction": allocation[
                        "quantized_parameter_fraction"
                    ],
                    "effective_bits_over_all_parameters": allocation[
                        "effective_bits_over_all_parameters"
                    ],
                    "evaluation_examples": sum(
                        int(item["num_examples"])
                        for item in record.metrics["evaluations"]
                    ),
                }
            )

    audit_payload = {
        "schema_version": 1,
        "runs_root": str(runs_root),
        "run_name_suffix": args.run_name_suffix,
        "manifest_root": str(manifest_root),
        "models": list(args.models),
        "arms": {
            arm: {
                "run_suffix": ARMS[arm][0],
                "frame_weighting": ARMS[arm][1],
            }
            for arm in ARM_ORDER
        },
        "datasets": {
            dataset: {
                "num_examples": len(ref_lengths[dataset]),
                "reference_words": int(ref_lengths[dataset].sum()),
                "manifest_sha256": sha256_file(manifest_root / f"{dataset}.json"),
            }
            for dataset in DATASETS
        },
        "bootstrap": {
            "unit": "utterance",
            "stratification": "dataset",
            "paired_cells": list(ARM_ORDER),
            "replicates": args.bootstrap_reps,
            "seed": args.seed,
            "ci": "percentile 95%",
            "p_value": "two-sided centered bootstrap with plus-one correction",
        },
        "primary_holm_family": [
            f"{model}:{contrast}"
            for model in args.models
            for contrast in PRIMARY_CONTRASTS
        ],
        "validations": validations,
    }

    write_csv(output_dir / "run_audit.csv", run_audit_rows)
    write_csv(output_dir / "wer_by_arm.csv", wer_rows)
    write_csv(output_dir / "bootstrap_contrasts.csv", contrast_rows)
    with (output_dir / "audit.json").open("w", encoding="utf-8") as handle:
        json.dump(audit_payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    report = build_report(
        runs,
        validations,
        wer_rows,
        contrast_rows,
        reps=args.bootstrap_reps,
        seed=args.seed,
    )
    (output_dir / "CP_ENCODER_BOOTSTRAP_REPORT.md").write_text(
        report, encoding="utf-8"
    )

    print(f"Audit passed for {len(args.models) * len(ARM_ORDER)} completed runs.")
    print(
        f"Analyzed {sum(len(ref_lengths[name]) for name in DATASETS):,} "
        f"utterances with {args.bootstrap_reps:,} paired replicates."
    )
    print(f"Wrote outputs to {output_dir}")


if __name__ == "__main__":
    main()
