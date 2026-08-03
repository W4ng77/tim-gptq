#!/usr/bin/env python3
"""Print state changes for an explicit list of experiment namespaces."""

from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path


def read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}
    return value if isinstance(value, dict) else {}


def snapshot(runs: Path, namespaces: list[str]) -> tuple[str, ...]:
    cells = []
    for namespace in namespaces:
        roots = sorted((runs / namespace).glob("*/status.json"))
        if not roots:
            cells.append(f"{namespace}:waiting:0")
            continue
        states = [read_json(path).get("state", "missing") for path in roots]
        evaluation_count = 0
        for status_path in roots:
            metrics = read_json(status_path.with_name("metrics.json"))
            evaluations = metrics.get("evaluations", [])
            evaluation_count += len(evaluations) if isinstance(evaluations, list) else 0
        state = states[0] if len(set(states)) == 1 else "+".join(states)
        cells.append(f"{namespace}:{state}:{evaluation_count}")
    return tuple(cells)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("namespace", nargs="+")
    parser.add_argument("--runs", type=Path, default=Path("runs"))
    parser.add_argument("--interval", type=int, default=30)
    args = parser.parse_args()
    previous: tuple[str, ...] | None = None
    while True:
        current = snapshot(args.runs, args.namespace)
        if current != previous:
            stamp = datetime.now(UTC).strftime("%F %T UTC")
            print(stamp, *current, flush=True)
            previous = current
        if current and all(":completed:" in cell for cell in current):
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
