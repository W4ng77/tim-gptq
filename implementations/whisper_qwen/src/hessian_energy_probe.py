"""Padding-Hessian energy probe for Whisper-style ASR calibration.

Quantifies how much of the GPTQ calibration statistics come from synthetic
padding: WhisperFeatureExtractor pads every clip to 3000 log-mel frames
(30 s), the conv front-end downsamples them to 1500 encoder positions, and
the per-example calibration loader keeps all of them. For each encoder layer
input (the rows feeding that layer's linear-group Gram matrices) and for the
final encoder hidden state (the rows feeding every decoder cross-attention
K/V projection during quantization capture), the probe reports:

- the padding-position fraction (by the feature attention mask, conv-aligned),
- the padding share of the Gram trace (trace(H) is proportional to the sum of
  squared row norms, so this is the padding contribution to Hessian energy),
- mean activation norms for speech vs padding positions and their ratio.

CPU-friendly at small nsamples. Full-size runs belong on the shared GPUs:

    CUDA_VISIBLE_DEVICES=4 python src/hessian_energy_probe.py \
        --model whisper-tiny --nsamples 128 --device cuda \
        --output-dir runs/hessian_probe

(only GPUs 4-7 may be used on this box; 0-3 are reserved).
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

from asr_data import get_asr_calibration_data
from asr_models import load_model, resolve_model_spec
from frame_weighting import downsample_feature_mask, whisper_encoder_conv_strides


def padding_energy_stats(hidden: torch.Tensor, position_mask: torch.Tensor) -> dict:
    """Accumulate row-energy statistics for one sample at one capture point.

    ``hidden`` is ``[positions, features]``; ``position_mask`` is
    ``[positions]`` with nonzero entries marking speech positions.
    """
    if hidden.ndim != 2:
        raise ValueError(f"Expected [positions, features], got {tuple(hidden.shape)}")
    if position_mask.numel() != hidden.shape[0]:
        raise ValueError(
            "Position mask does not align with hidden positions: "
            f"{position_mask.numel()} != {hidden.shape[0]}"
        )
    squared = hidden.detach().float().square().sum(dim=-1)
    norms = squared.sqrt()
    speech = position_mask.detach().reshape(-1).bool()
    padding = ~speech
    return {
        "positions": int(squared.numel()),
        "padding_positions": int(padding.sum()),
        "sum_sq_total": float(squared.sum()),
        "sum_sq_padding": float(squared[padding].sum()),
        "norm_sum_speech": float(norms[speech].sum()),
        "norm_sum_padding": float(norms[padding].sum()),
    }


def merge_stats(accumulator: dict, update: dict) -> dict:
    for key, value in update.items():
        accumulator[key] = accumulator.get(key, 0) + value
    return accumulator


def finalize_stats(stats: dict) -> dict:
    positions = max(int(stats.get("positions", 0)), 1)
    padding_positions = int(stats.get("padding_positions", 0))
    speech_positions = positions - padding_positions
    sum_sq_total = float(stats.get("sum_sq_total", 0.0))
    mean_norm_speech = stats.get("norm_sum_speech", 0.0) / max(speech_positions, 1)
    mean_norm_padding = stats.get("norm_sum_padding", 0.0) / max(padding_positions, 1)
    return {
        "positions": positions,
        "padding_positions": padding_positions,
        "padding_position_fraction": padding_positions / positions,
        "gram_trace_total": sum_sq_total,
        "padding_gram_trace_fraction": (
            stats.get("sum_sq_padding", 0.0) / sum_sq_total
            if sum_sq_total > 0.0
            else 0.0
        ),
        "mean_norm_speech": mean_norm_speech,
        "mean_norm_padding": mean_norm_padding,
        "speech_to_padding_norm_ratio": (
            mean_norm_speech / mean_norm_padding if mean_norm_padding > 0.0 else float("inf")
        ),
    }


def build_probe_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Measure the padding-frame contribution to encoder / cross-attn "
            "K/V calibration Hessian energy."
        )
    )
    parser.add_argument("--model", default="whisper-tiny")
    parser.add_argument("--nsamples", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", default="runs/hessian_probe")
    return parser


@torch.no_grad()
def run_probe(args):
    spec = resolve_model_spec(args.model)
    if spec.family != "whisper":
        raise ValueError(
            "The padding-Hessian probe targets Whisper's fixed-length padded "
            f"features; model family {spec.family!r} is out of scope "
            "(Moonshine input_values are variable-length and unpadded)."
        )

    model, processor = load_model(spec)
    device = torch.device(args.device)
    model = model.to(device)
    dtype = next(iter(model.parameters())).dtype
    strides = whisper_encoder_conv_strides(model)

    calibration_data = get_asr_calibration_data(
        processor,
        nsamples=args.nsamples,
        seed=args.seed,
        offset=args.offset,
        include_feature_attention_mask=True,
    )

    encoder = model.model.encoder
    num_layers = len(encoder.layers)
    layer_stats = [dict() for _ in range(num_layers)]
    cross_stats: dict = {}

    for sample in calibration_data:
        features = sample["input_features"].to(device=device, dtype=dtype)
        mask = sample.get("feature_attention_mask")
        if mask is None:
            raise RuntimeError(
                "Missing feature_attention_mask; the processor did not return "
                "a frame-level attention mask."
            )
        position_mask = downsample_feature_mask(mask, strides).reshape(-1)
        output = encoder(features, output_hidden_states=True)
        hidden_states = output.hidden_states
        # hidden_states[i] is the input to encoder layer i; the final entry is
        # the post-layer-norm encoder output, which is exactly what the
        # decoder cross-attention k_proj/v_proj hooks capture during
        # quantization (run_decoder_layer feeds encoder_outputs[sample_idx]).
        for index in range(num_layers):
            merge_stats(
                layer_stats[index],
                padding_energy_stats(hidden_states[index][0], position_mask),
            )
        merge_stats(
            cross_stats,
            padding_energy_stats(output.last_hidden_state[0], position_mask),
        )

    rows = []
    for index in range(num_layers):
        rows.append(
            {
                "scope": "enc",
                "layer": index,
                "capture": "encoder_layer_input",
                **finalize_stats(layer_stats[index]),
            }
        )
    rows.append(
        {
            "scope": "dec",
            "layer": "all",
            "capture": "cross_attn_kv_input",
            **finalize_stats(cross_stats),
        }
    )

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{spec.alias}__n{args.nsamples}__seed{args.seed}__off{args.offset}"
    csv_path = output_dir / f"{stem}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    encoder_fractions = [
        row["padding_gram_trace_fraction"] for row in rows if row["scope"] == "enc"
    ]
    summary = {
        "model": spec.alias,
        "model_id": spec.model_id,
        "nsamples": int(args.nsamples),
        "seed": int(args.seed),
        "offset": int(args.offset),
        "device": str(args.device),
        "conv_strides": list(strides),
        "mean_padding_position_fraction": sum(
            row["padding_position_fraction"] for row in rows
        )
        / len(rows),
        "encoder_padding_gram_trace_fraction_mean": (
            sum(encoder_fractions) / len(encoder_fractions)
        ),
        "encoder_padding_gram_trace_fraction_first_layer": encoder_fractions[0],
        "encoder_padding_gram_trace_fraction_last_layer": encoder_fractions[-1],
        "cross_attn_kv_padding_gram_trace_fraction": rows[-1][
            "padding_gram_trace_fraction"
        ],
        "cross_attn_kv_speech_to_padding_norm_ratio": rows[-1][
            "speech_to_padding_norm_ratio"
        ],
        "csv": str(csv_path),
        "gpu_command_full_calibration": (
            "CUDA_VISIBLE_DEVICES=4 python src/hessian_energy_probe.py "
            f"--model {spec.alias} --nsamples 128 --device cuda "
            "--output-dir runs/hessian_probe  # GPUs 4-7 only"
        ),
    }
    json_path = output_dir / f"{stem}.json"
    json_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(f"[probe] wrote {csv_path}")
    print(f"[probe] wrote {json_path}")
    for row in rows:
        print(
            f"[probe] {row['scope']}.{row['layer']:>3} {row['capture']:>22} "
            f"pad_pos={row['padding_position_fraction']:.3f} "
            f"pad_trace={row['padding_gram_trace_fraction']:.3f} "
            f"norm_speech={row['mean_norm_speech']:.2f} "
            f"norm_pad={row['mean_norm_padding']:.2f} "
            f"speech/pad={row['speech_to_padding_norm_ratio']:.2f}"
        )
    return csv_path, summary


def main() -> None:
    args = build_probe_parser().parse_args()
    run_probe(args)


if __name__ == "__main__":
    main()
