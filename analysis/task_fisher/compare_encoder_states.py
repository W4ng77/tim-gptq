#!/usr/bin/env python
"""Compare exported encoder state artifacts without materializing a model."""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path

import torch


def load_state(path: Path) -> dict[str, torch.Tensor]:
    payload = torch.load(
        path, map_location="cpu", weights_only=True, mmap=True
    )
    state = payload.get("state_dict", payload)
    if not isinstance(state, dict):
        raise TypeError(f"{path}: expected a state_dict mapping")
    return state


def compare(
    left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]
) -> dict:
    keys = sorted(set(left) & set(right))
    total = different = 0
    abs_sum = diff_sq = left_sq = right_sq = 0.0
    per_tensor = []

    for key in keys:
        a, b = left[key], right[key]
        if not (
            isinstance(a, torch.Tensor)
            and isinstance(b, torch.Tensor)
            and a.shape == b.shape
            and (a.is_floating_point() or a.is_complex())
        ):
            continue
        a = a.detach()
        b = b.detach()
        d = a.float() - b.float()
        n = d.numel()
        n_diff = int(torch.count_nonzero(a != b).item())
        dsq = float(torch.sum(d * d, dtype=torch.float64).item())
        asq = float(torch.sum(a.float() ** 2, dtype=torch.float64).item())
        bsq = float(torch.sum(b.float() ** 2, dtype=torch.float64).item())

        total += n
        different += n_diff
        abs_sum += float(torch.sum(torch.abs(d), dtype=torch.float64).item())
        diff_sq += dsq
        left_sq += asq
        right_sq += bsq
        denom = 0.5 * (asq + bsq)
        per_tensor.append(
            {
                "name": key,
                "numel": n,
                "different_fraction": n_diff / n,
                "symmetric_relative_l2": math.sqrt(dsq / denom)
                if denom
                else 0.0,
            }
        )

    denom = 0.5 * (left_sq + right_sq)
    return {
        "common_floating_tensors": len(per_tensor),
        "numel": total,
        "different_numel": different,
        "different_fraction": different / total,
        "mean_absolute_delta": abs_sum / total,
        "symmetric_relative_l2": math.sqrt(diff_sq / denom) if denom else 0.0,
        "top_tensors_by_relative_l2": sorted(
            per_tensor,
            key=lambda row: row["symmetric_relative_l2"],
            reverse=True,
        )[:20],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--state",
        nargs=2,
        metavar=("LABEL", "PATH"),
        action="append",
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    torch.set_num_threads(8)
    states = {
        label: load_state(Path(path)) for label, path in args.state
    }
    report = {
        f"{left}__vs__{right}": compare(states[left], states[right])
        for left, right in itertools.combinations(states, 2)
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
