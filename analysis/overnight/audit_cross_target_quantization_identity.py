#!/usr/bin/env python3
"""Audit whether cross-target runs repeat the same quantization computation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


EVAL_ONLY_FIELDS = {"datasets", "eval_manifest_dir", "output_dir", "run_name"}


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def file_sha(path: Path) -> str | None:
    return sha256_bytes(path.read_bytes()) if path.is_file() else None


def canonical_config_sha(path: Path, missing_false_fields: tuple[str, ...]) -> str | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    canonical = {key: value for key, value in payload.items() if key not in EVAL_ONLY_FIELDS}
    for field in missing_false_fields:
        canonical.setdefault(field, False)
    return sha256_bytes(
        json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pair", nargs=4, action="append", required=True,
        metavar=("ARM", "DRAW", "TARGET_A_RUN", "TARGET_B_RUN"),
    )
    parser.add_argument("--target-labels", nargs=2, required=True)
    parser.add_argument(
        "--normalize-missing-false",
        action="append",
        default=[],
        metavar="FIELD",
        help=(
            "Treat an absent optional boolean as its explicit false default; "
            "every normalized field is recorded in the audit report."
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    for arm, draw, left_text, right_text in args.pair:
        left, right = Path(left_text), Path(right_text)
        calibration = (file_sha(left / "calibration.json"), file_sha(right / "calibration.json"))
        quantization = (file_sha(left / "quantization.json"), file_sha(right / "quantization.json"))
        missing_false_fields = tuple(args.normalize_missing_false)
        config = (
            canonical_config_sha(left / "config.json", missing_false_fields),
            canonical_config_sha(right / "config.json", missing_false_fields),
        )
        complete = all(value is not None for pair in (calibration, quantization, config) for value in pair)
        matched = complete and all(a == b for a, b in (calibration, quantization, config))
        rows.append({
            "arm": arm,
            "draw": draw,
            "target_a_run": str(left),
            "target_b_run": str(right),
            "calibration_sha256": list(calibration),
            "quantization_sha256": list(quantization),
            "canonical_config_sha256": list(config),
            "complete": complete,
            "matched": matched,
        })

    payload = {
        "target_labels": args.target_labels,
        "canonical_config_removed_fields": sorted(EVAL_ONLY_FIELDS),
        "canonical_config_missing_false_normalizations": sorted(
            args.normalize_missing_false
        ),
        "all_complete": all(row["complete"] for row in rows),
        "all_matched": all(row["matched"] for row in rows),
        "rows": rows,
        "limitation": (
            "Protocol identity only: full post-quantization parameter tensors were not "
            "serialized and therefore cannot be byte-hashed."
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "cross_target_identity_audit.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = [
        "# Cross-target quantization-identity audit",
        "",
        f"Targets: `{args.target_labels[0]}` and `{args.target_labels[1]}`. "
        f"All complete: `{payload['all_complete']}`; all matched: `{payload['all_matched']}`.",
        "",
        "Canonical config removes only evaluation/artifact fields: `" +
        "`, `".join(sorted(EVAL_ONLY_FIELDS)) + "`.",
        (
            "Missing optional booleans normalized to explicit false: `" +
            "`, `".join(sorted(args.normalize_missing_false)) + "`."
            if args.normalize_missing_false
            else "No missing-field default normalization was applied."
        ),
        "",
        "| Arm | Draw | Calibration | Quantization | Canonical config | Complete |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        symbols = []
        for key in ("calibration_sha256", "quantization_sha256", "canonical_config_sha256"):
            a, b = row[key]
            symbols.append("✓" if a is not None and a == b else ("missing" if None in (a, b) else "✗"))
        lines.append(
            f"| {row['arm']} | {row['draw']} | {symbols[0]} | {symbols[1]} | "
            f"{symbols[2]} | {row['complete']} |"
        )
    lines.extend(["", payload["limitation"]])
    (args.output_dir / "CROSS_TARGET_IDENTITY_AUDIT.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )

    if payload["all_complete"] and not payload["all_matched"]:
        raise SystemExit("Completed cross-target artifacts do not match")


if __name__ == "__main__":
    main()
