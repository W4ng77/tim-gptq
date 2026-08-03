"""Unified experiment CLI for Qwen3-ASR, Moonshine, and Whisper."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import platform
import random
import subprocess
import time
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import datasets
import numpy as np
import torch
import transformers

from asr_data import (
    CALIBRATION_CONFIG,
    CALIBRATION_DATASET_ID,
    CALIBRATION_SPLIT,
    get_asr_calibration_data,
)
from asr_eval import (
    DATASET_SPECS,
    EVALUATION_DATASET_ID,
    evaluate_wer,
    score_calibration_nll,
    score_qwen_calibration_wer,
    score_calibration_wer,
)
from asr_models import MODEL_SPECS, load_model, resolve_model_spec
from quantization_runtime import (
    DEV,
    _move_model_to_cuda_if_possible,
    _qwen_group_key_from_module_name,
    _qwen_scope_includes_stack,
    _qwen_collect_audio_task_fisher_weights,
    _qwen_collect_sequence_token_weights,
    _qwen_make_calibration_data,
    _qwen_sequence_support_masks,
    _qwen_quantize_cached_sequential,
    _set_eval_mode,
    _supports_asr_pipeline,
    iter_rtn_named_modules,
    rtn_quantize_model_inplace,
)
from asr_quantization import (
    QUANT_SCOPES,
    collect_decoder_propagation_token_weights,
    collect_decoder_sequence_token_weights,
    collect_fp_encoder_outputs,
    quantize_decoder_with_dynamic_alpha,
    quantize_decoder_with_alpha,
    quantize_encoder_with_alpha,
    quantize_with_dynamic_alpha,
)
from frame_weighting import (
    DEFAULT_ATTENTION_FLOOR,
    DEFAULT_PROPAGATED_PROBES,
    PROPAGATED_CLIP_MAX,
    PROPAGATED_CLIP_MIN,
    compute_calibration_frame_weights,
    finalize_frame_weight_statistics,
    permute_calibration_token_weights,
    permute_calibration_token_weights_stratified,
)
from quantization_utils import map_scores_to_alpha
from interface_bridge import fit_affine_encoder_bridge
from sequence_objective import (
    paired_example_bootstrap_ci,
    paired_nll_recovery,
    paired_transcript_bootstrap_ci,
    paired_transcript_drift_recovery,
    select_best_sequence_candidate,
)
from quantization_utils import _record_fp16_protected_module


MODES = (
    "fp16",
    "rtn",
    "awq",
    "gptq",
    "gptq+gptaq",
    "gptq+qep",
    "gptq+dynamic-alpha",
    "gptq+ffncomp",
    "gptq+ffngate",
    "gptq+ffnprotect",
    "gptq+tailffnprotect",
    "gptq+seqprotect",
    "gptq+seqcal",
    "gptq+seqhess",
    "gptq+prophess",
    "gptq+boundaryguard",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run matched ASR quantization experiments across supported model families."
    )
    parser.add_argument("--model", default="whisper-tiny", help="Model alias or model id.")
    parser.add_argument("--list-models", action="store_true")
    parser.add_argument("--mode", choices=MODES, default="fp16")
    parser.add_argument(
        "--quant-scope",
        choices=QUANT_SCOPES,
        default="full",
        help="Component scope; non-full scopes currently target encoder-decoder models.",
    )
    parser.add_argument(
        "--sequence-support",
        choices=("full", "prompt", "full-matched"),
        default="full",
        help=(
            "Qwen text-backbone calibration rows: 'full' uses the complete "
            "teacher-forced prompt+transcript trajectory; 'prompt' computes "
            "any task-density score from the complete transcript loss but "
            "restricts the GPTQ Gram to prompt-prefix states; 'full-matched' "
            "draws the same number of rows from the full trajectory."
        ),
    )
    parser.add_argument(
        "--sequence-support-seed",
        type=int,
        default=-1,
        help=(
            "Seed for full-matched row selection; negative reuses the "
            "resolved quantization seed."
        ),
    )
    parser.add_argument(
        "--interface-bridge",
        choices=("none", "affine"),
        default="none",
        help="Calibration-only encoder-interface intervention for P2 diagnosis.",
    )
    parser.add_argument(
        "--collect-context-scores",
        action="store_true",
        help="Record encoder error weighted by decoder cross-attention K/V sensitivity.",
    )
    parser.add_argument(
        "--score-calibration-nll",
        action="store_true",
        help="Record teacher-forced calibration token NLL after quantization.",
    )
    parser.add_argument(
        "--score-calibration-wer",
        action="store_true",
        help="Record end-to-end WER on the quantization calibration examples.",
    )
    parser.add_argument(
        "--calibration-score-samples",
        type=int,
        default=0,
        help="Use this many held-out calibration examples for scoring; zero reuses fit data.",
    )
    parser.add_argument(
        "--calibration-score-offset",
        type=int,
        default=0,
        help="Offset in the deterministic calibration order for held-out scoring.",
    )
    parser.add_argument(
        "--calibration-score-config",
        default=CALIBRATION_CONFIG,
        help=(
            "Dataset config used only for held-out calibration scoring; "
            "quantizer fitting remains on the registered calibration config."
        ),
    )
    parser.add_argument(
        "--calibration-score-split",
        default=CALIBRATION_SPLIT,
        help=(
            "Dataset split used only for held-out calibration scoring; "
            "for example other/train.500."
        ),
    )
    parser.add_argument(
        "--calibration-score-augment",
        choices=("none", "acoustic"),
        default="none",
        help=(
            "Apply deterministic waveform stress augmentation only to the "
            "held-out calibration-scoring split; quantizer fitting data are "
            "unchanged."
        ),
    )
    parser.add_argument(
        "--calibration-score-augment-ratio",
        type=float,
        default=1.0,
        help=(
            "Fraction of held-out scoring examples receiving "
            "--calibration-score-augment acoustic."
        ),
    )
    parser.add_argument(
        "--boundaryguard-min-calibration-wer-improvement",
        type=float,
        default=0.02,
        help="Minimum absolute calibration-WER gain required to enable protection.",
    )
    parser.add_argument(
        "--boundaryguard-min-relative-nll-degradation",
        type=float,
        default=0.10,
        help=(
            "Enable protection when uniform quantization degrades calibration NLL "
            "by at least this fraction and protection lowers that NLL."
        ),
    )
    parser.add_argument("--wbits", type=int, choices=(2, 3, 4), default=4)
    parser.add_argument(
        "--groupsize",
        type=int,
        default=0,
        help="Quantization group size. Zero selects the registry default.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--quantization-seed",
        type=int,
        default=-1,
        help=(
            "RNG seed for model loading and quantization. Negative reuses --seed, "
            "which remains the calibration-example seed."
        ),
    )
    parser.add_argument("--nsamples", type=int, default=128)
    parser.add_argument("--calib-batch-size", type=int, default=16)
    parser.add_argument(
        "--frame-weighting",
        choices=(
            "none",
            "mask",
            "energy",
            "attention",
            "propagated-uniform",
            "propagated",
            "task-fisher",
        ),
        default="none",
        help=(
            "Row-weight encoder-position calibration statistics (encoder "
            "groups and decoder cross-attention K/V): 'mask' zeroes Whisper "
            "padding frames, 'energy' weights frames by mel energy, "
            "'attention' weights frames by FP16 teacher-forced decoder "
            "cross-attention mass (soft, floored; padding stays nonzero to "
            "match inference-time consumption), 'propagated-uniform' pulls "
            "an identity terminal metric back through the encoder Jacobian, "
            "and 'propagated' pulls the same attention metric back. The last "
            "two modes separate layer transport from decoder consumption "
            "(see --propagated-probes). 'task-fisher' instead uses the "
            "label-conditioned squared task-loss gradient at every encoder "
            "layer input. mask/energy are no-ops for "
            "Moonshine; attention/propagated also weight Moonshine; every "
            "attention/propagated remain recorded no-ops for Qwen, while "
            "task-fisher weights Qwen audio-tower rows directly. Requires "
            "a GPTQ-derived mode without dual-stream statistics (QEP/GPTAQ)."
        ),
    )
    parser.add_argument(
        "--frame-weighting-floor",
        type=float,
        default=DEFAULT_ATTENTION_FLOOR,
        help=(
            "Minimum terminal weight for attention and propagated modes. "
            "The softly floored weights are renormalized to unit mean."
        ),
    )
    parser.add_argument(
        "--propagated-probes",
        type=int,
        default=DEFAULT_PROPAGATED_PROBES,
        help=(
            "Rademacher VJP probes per calibration sample for "
            "--frame-weighting propagated or propagated-uniform "
            "(encoder-side Hutchinson estimate of the metric pullback)."
        ),
    )
    parser.add_argument(
        "--propagated-clip-max",
        type=float,
        default=PROPAGATED_CLIP_MAX,
        help=(
            "Robust upper clip for unit-mean per-layer pullback weights in "
            "propagated modes; those weights are renormalized after "
            "clipping. Task-Fisher instead uses a ratio-preserving KL/I-"
            "projection to unit mean while keeping the returned weights "
            "inside the final bounds."
        ),
    )
    parser.add_argument(
        "--task-fisher-min-ess-fraction",
        type=float,
        default=0.0,
        help=(
            "Minimum effective-frame fraction for task-Fisher row weights. "
            "Analytically shrink toward uniform just enough to satisfy this "
            "ESS constraint; zero disables ESS shrinkage."
        ),
    )
    parser.add_argument(
        "--task-weight-permutation",
        choices=("none", "within-sample", "within-prompt-response"),
        default="none",
        help=(
            "Identification control that permutes final bounded task weights "
            "within each sample/layer. The prompt-response mode permutes "
            "separately inside the two label-defined strata."
        ),
    )
    parser.add_argument(
        "--task-weight-permutation-seed",
        type=int,
        default=-1,
        help=(
            "Seed for --task-weight-permutation; negative reuses the resolved "
            "quantization seed."
        ),
    )
    parser.add_argument(
        "--calib-augment",
        choices=("none", "acoustic"),
        default="none",
        help=(
            "Deterministic waveform-level calibration augmentation "
            "(speed 0.9/1.1, gain +/-6 dB, white noise SNR 5-20 dB, synthetic "
            "reverb), chosen per (seed, example index)."
        ),
    )
    parser.add_argument(
        "--calib-augment-ratio",
        type=float,
        default=0.5,
        help="Fraction of calibration examples receiving acoustic augmentation.",
    )
    parser.add_argument("--percdamp", type=float, default=0.01)
    parser.add_argument("--percdampqep", type=float, default=1.0)
    parser.add_argument("--perccorr", type=float, default=0.5)
    parser.add_argument("--act-order", action="store_true")
    parser.add_argument(
        "--gptaq-alpha",
        type=float,
        default=0.25,
        help=(
            "Scale of the GPTAQ asymmetric-calibration correction "
            "(arXiv:2504.02692); used by mode gptq+gptaq."
        ),
    )
    parser.add_argument(
        "--rotate",
        choices=("none", "hadamard"),
        default="none",
        help=(
            "Quantize each linear in a randomized block-diagonal Hadamard "
            "basis and fold back (QuaRot/QuIP-style, fake-quant equivalent). "
            "Requires a GPTQ-derived mode."
        ),
    )
    parser.add_argument(
        "--rotation-seed",
        type=int,
        default=-1,
        help=(
            "RNG seed for the random Hadamard sign vectors. Negative reuses "
            "--quantization-seed."
        ),
    )
    parser.add_argument(
        "--cross-attn-wbits",
        type=int,
        choices=(0, 3, 4),
        default=0,
        help="Override decoder cross-attention weight bits; zero uses --wbits.",
    )
    parser.add_argument(
        "--encoder-tail-wbits",
        type=int,
        choices=(0, 3, 4),
        default=0,
        help="Override the final encoder blocks' weight bits; zero uses --wbits.",
    )
    parser.add_argument(
        "--encoder-tail-layers",
        type=int,
        default=0,
        help="Number of final encoder blocks receiving --encoder-tail-wbits.",
    )
    parser.add_argument(
        "--encoder-promote-layers",
        default="",
        help="Comma-separated encoder layer indices receiving --encoder-tail-wbits.",
    )
    parser.add_argument("--alpha-min", type=float, default=0.1)
    parser.add_argument("--alpha-max", type=float, default=0.8)
    parser.add_argument(
        "--tail-ffn-fraction",
        type=float,
        default=0.25,
        help="Final block fraction protected by gptq+tailffnprotect.",
    )
    parser.add_argument(
        "--propagation-hessian-probes",
        type=int,
        default=2,
        help="Rademacher VJPs per layer for one-block propagated curvature.",
    )
    parser.add_argument(
        "--sequence-objective",
        choices=("nll", "fp-transcript"),
        default="nll",
        help=(
            "Sequence routing objective: ground-truth teacher-forced NLL or "
            "autoregressive transcript drift from the FP16 model."
        ),
    )
    parser.add_argument(
        "--sequence-selection-samples",
        type=int,
        default=50,
        help="Disjoint examples used to rank per-layer FP16 restorations.",
    )
    parser.add_argument(
        "--sequence-confirmation-samples",
        type=int,
        default=50,
        help="Disjoint examples used to confirm the selected restoration.",
    )
    parser.add_argument(
        "--sequence-bootstrap-replicates",
        type=int,
        default=10_000,
        help="Paired example-bootstrap replicates for sequence confirmation.",
    )
    parser.add_argument(
        "--sequence-confidence",
        type=float,
        default=0.95,
        help="Two-sided confidence level for sequence confirmation.",
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Run WER evaluation after optional quantization.",
    )
    parser.add_argument(
        "--datasets",
        default="all",
        help="Comma-separated evaluation aliases, or 'all' for every registered split.",
    )
    parser.add_argument(
        "--eval-samples",
        type=int,
        default=-1,
        help="Examples per dataset; -1 evaluates the complete split.",
    )
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument(
        "--eval-manifest-dir",
        default="",
        help="Directory containing one fixed <dataset-alias>.json selection manifest.",
    )
    parser.add_argument("--output-dir", default="runs")
    parser.add_argument("--run-name", default="")
    parser.add_argument(
        "--save-encoder-state",
        default="",
        help=(
            "After optional quantization, export the Whisper encoder state and "
            "a reproducibility sidecar to this path. Existing files are never "
            "overwritten."
        ),
    )
    parser.add_argument(
        "--load-encoder-state",
        default="",
        help=(
            "Strictly load a previously exported Whisper encoder before "
            "evaluation. This requires --mode fp16 so the imported encoder "
            "cannot be quantized a second time; run metadata labels the model "
            "as an imported-encoder state rather than an FP16 baseline."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print the resolved configuration without loading a model.",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser


def resolve_mode(args) -> None:
    args.method = "fp16" if args.mode == "fp16" else args.mode.split("+", 1)[0]
    args.qep = args.mode in {
        "gptq+qep",
        "gptq+dynamic-alpha",
        "gptq+ffncomp",
        "gptq+ffngate",
    }
    args.qep_target = (
        "ffn-output"
        if args.mode in {"gptq+ffncomp", "gptq+ffngate"}
        else "legacy-except-fc2"
    )
    args.qep_gate = args.mode == "gptq+ffngate"
    args.qep_gate_candidates = (0.0, 0.1, 0.25, 0.5)
    args.gptaq = args.mode == "gptq+gptaq"
    args.ffn_output_fp16 = args.mode == "gptq+ffnprotect"
    args.tail_ffn_output_fp16 = args.mode == "gptq+tailffnprotect"
    args.sequence_protect = args.mode == "gptq+seqprotect"
    args.sequence_calibration = args.mode == "gptq+seqcal"
    args.sequence_hessian = args.mode == "gptq+seqhess"
    args.propagation_hessian = args.mode == "gptq+prophess"
    args.dynamic_alpha = args.mode == "gptq+dynamic-alpha"
    args.boundaryguard = args.mode == "gptq+boundaryguard"
    args.int_zero_point = True
    args.encoder_promote_layer_indices = tuple(
        sorted(
            {
                int(item.strip())
                for item in args.encoder_promote_layers.split(",")
                if item.strip()
            }
        )
    )


def resolve_run_seeds(args) -> None:
    """Keep calibration sampling and quantization RNG independently auditable."""
    if args.quantization_seed < 0:
        args.quantization_seed = args.seed
    if getattr(args, "rotation_seed", -1) < 0:
        args.rotation_seed = args.quantization_seed


def parse_datasets(raw: str) -> list[str]:
    if raw.strip().lower() == "all":
        return list(DATASET_SPECS)
    datasets = [item.strip() for item in raw.split(",") if item.strip()]
    unknown = sorted(set(datasets) - set(DATASET_SPECS))
    if unknown:
        raise ValueError(
            f"Unknown datasets {unknown}; supported: {sorted(DATASET_SPECS)}"
        )
    if not datasets:
        raise ValueError("At least one evaluation dataset is required.")
    return datasets


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _source_tree_sha256() -> str:
    """Hash active experiment sources, including uncommitted and untracked files."""
    root = Path(__file__).resolve().parents[1]
    files = []
    for directory in ("src", "scripts", "tests"):
        files.extend(
            path
            for path in (root / directory).rglob("*")
            if path.is_file() and "__pycache__" not in path.parts
        )
    for name in ("README.md", "requirements.txt"):
        path = root / name
        if path.is_file():
            files.append(path)

    digest = hashlib.sha256()
    for path in sorted(set(files)):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        content = path.read_bytes()
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def _environment_record() -> dict:
    try:
        qwen_asr_version = version("qwen-asr")
    except PackageNotFoundError:
        qwen_asr_version = "not-installed"
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "datasets": datasets.__version__,
        "qwen_asr": qwen_asr_version,
        "cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu_names": [
            torch.cuda.get_device_name(index)
            for index in range(torch.cuda.device_count())
        ],
        "conda_prefix": os.environ.get("CONDA_PREFIX", ""),
        "hf_home": os.environ.get("HF_HOME", ""),
        "tmpdir": os.environ.get("TMPDIR", ""),
        "git_commit": _git_commit(),
        "source_tree_sha256": _source_tree_sha256(),
    }


def _model_parameter_stats(model) -> dict:
    owner = model if isinstance(model, torch.nn.Module) else getattr(model, "model", None)
    if not isinstance(owner, torch.nn.Module):
        return {}
    total = sum(int(parameter.numel()) for parameter in owner.parameters())
    linear_weights = sum(
        int(module.weight.numel())
        for module in owner.modules()
        if isinstance(module, torch.nn.Linear)
    )
    return {
        "total_parameter_numel": total,
        "linear_weight_numel": linear_weights,
    }


def _create_run_dir(args, model_alias: str) -> Path:
    if args.run_name:
        run_name = args.run_name
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        state_label = (
            "encoder-import"
            if getattr(args, "load_encoder_state", "")
            else args.mode
        )
        run_name = (
            f"{stamp}__{model_alias}__{state_label}__w{args.wbits}"
            f"__g{args.groupsize}__seed{args.seed}"
        )
    run_dir = Path(args.output_dir).expanduser().resolve() / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _write_json(path: Path, payload) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


_WHISPER_ENCODER_STATE_FORMAT = "tim_gptq.whisper_encoder_state"
_WHISPER_ENCODER_STATE_VERSION = 1
_WHISPER_ENCODER_CONFIG_KEYS = (
    "activation_dropout",
    "activation_function",
    "attention_dropout",
    "d_model",
    "dropout",
    "encoder_attention_heads",
    "encoder_ffn_dim",
    "encoder_layerdrop",
    "encoder_layers",
    "init_std",
    "max_source_positions",
    "num_mel_bins",
    "scale_embedding",
)


def _encoder_state_sidecar_path(path: Path) -> Path:
    return path.with_name(f"{path.name}.metadata.json")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _whisper_encoder(model) -> torch.nn.Module:
    backbone = getattr(model, "model", None)
    encoder = getattr(backbone, "encoder", None)
    if not isinstance(encoder, torch.nn.Module):
        raise ValueError(
            "Encoder state exchange requires a Whisper model exposing "
            "model.encoder as a torch.nn.Module."
        )
    return encoder


def _whisper_encoder_config(model) -> dict:
    """Return the strict encoder-relevant config shared by Large and Distil."""
    encoder = _whisper_encoder(model)
    config = getattr(model, "config", None)
    if config is None:
        config = getattr(encoder, "config", None)
    if config is None:
        raise ValueError("Whisper encoder has no configuration to validate.")

    values = {}
    missing = []
    for key in _WHISPER_ENCODER_CONFIG_KEYS:
        if not hasattr(config, key):
            missing.append(key)
            continue
        value = getattr(config, key)
        try:
            json.dumps(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Whisper encoder config field {key!r} is not JSON-serializable."
            ) from exc
        values[key] = value
    if missing:
        raise ValueError(
            "Whisper encoder config is missing required fields: "
            + ", ".join(missing)
        )
    return {
        "encoder_class": (
            f"{type(encoder).__module__}.{type(encoder).__qualname__}"
        ),
        "fields": values,
    }


def _state_dict_manifest(state_dict: dict[str, torch.Tensor]) -> dict:
    manifest = {}
    for name in sorted(state_dict):
        tensor = state_dict[name]
        if not isinstance(name, str) or not isinstance(tensor, torch.Tensor):
            raise ValueError(
                "Whisper encoder state_dict must map string names to tensors."
            )
        manifest[name] = {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "numel": int(tensor.numel()),
        }
    return manifest


def _state_dict_sha256(state_dict: dict[str, torch.Tensor]) -> str:
    """Hash names, schemas, and exact tensor bytes in a device-independent order."""
    digest = hashlib.sha256()
    for name in sorted(state_dict):
        tensor = state_dict[name]
        if not isinstance(name, str) or not isinstance(tensor, torch.Tensor):
            raise ValueError(
                "Whisper encoder state_dict must map string names to tensors."
            )
        contiguous = tensor.detach().cpu().contiguous()
        name_bytes = name.encode("utf-8")
        dtype_bytes = str(contiguous.dtype).encode("ascii")
        shape_bytes = json.dumps(list(contiguous.shape)).encode("ascii")
        raw = contiguous.view(torch.uint8).numpy().tobytes()
        for part in (name_bytes, dtype_bytes, shape_bytes, raw):
            digest.update(len(part).to_bytes(8, "big"))
            digest.update(part)
    return digest.hexdigest()


def _validate_encoder_state_args(args, spec) -> None:
    save_path = str(getattr(args, "save_encoder_state", "") or "").strip()
    load_path = str(getattr(args, "load_encoder_state", "") or "").strip()
    if save_path and load_path:
        raise ValueError(
            "--save-encoder-state and --load-encoder-state are mutually exclusive."
        )
    if not save_path and not load_path:
        args.effective_model_state = (
            "native-fp16"
            if args.mode == "fp16"
            else f"{args.mode}:{args.quant_scope}"
        )
        return
    if spec.family != "whisper":
        raise ValueError(
            "Encoder state exchange is supported only for Whisper-family models."
        )

    if save_path:
        args.save_encoder_state = str(Path(save_path).expanduser().resolve())
        args.effective_model_state = (
            "exported-native-fp16-encoder"
            if args.mode == "fp16"
            else f"exported-{args.mode}-{args.quant_scope}-encoder"
        )
        return

    if args.mode != "fp16":
        raise ValueError(
            "--load-encoder-state requires --mode fp16. The mode is reserved "
            "as a no-op execution path so an imported, possibly quantized "
            "encoder cannot be quantized a second time."
        )
    args.load_encoder_state = str(Path(load_path).expanduser().resolve())
    args.effective_model_state = "fp16-consumer-with-imported-encoder"


def _encoder_state_source_metadata(spec, args) -> dict:
    return {
        "model_alias": spec.alias,
        "model_id": spec.model_id,
        "model_family": spec.family,
        "mode": args.mode,
        "quant_scope": args.quant_scope,
        "wbits": int(args.wbits),
        "groupsize": int(args.groupsize),
        "seed": int(args.seed),
        "quantization_seed": int(args.quantization_seed),
        "frame_weighting": str(args.frame_weighting),
        "frame_weighting_floor": float(args.frame_weighting_floor),
        "propagated_probes": int(args.propagated_probes),
        "git_commit": _git_commit(),
        "source_tree_sha256": _source_tree_sha256(),
        "transformers_version": transformers.__version__,
        "torch_version": str(torch.__version__),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def _export_whisper_encoder_state(
    model,
    path: str | Path,
    spec,
    args,
) -> dict:
    artifact_path = Path(path).expanduser().resolve()
    sidecar_path = _encoder_state_sidecar_path(artifact_path)
    if artifact_path.exists() or sidecar_path.exists():
        raise FileExistsError(
            "Refusing to overwrite encoder state artifact or sidecar: "
            f"{artifact_path}, {sidecar_path}"
        )
    artifact_path.parent.mkdir(parents=True, exist_ok=True)

    encoder = _whisper_encoder(model)
    state_dict = {
        name: tensor.detach().cpu().clone()
        for name, tensor in encoder.state_dict().items()
    }
    manifest = _state_dict_manifest(state_dict)
    state_sha256 = _state_dict_sha256(state_dict)
    source = _encoder_state_source_metadata(spec, args)
    payload = {
        "format": _WHISPER_ENCODER_STATE_FORMAT,
        "format_version": _WHISPER_ENCODER_STATE_VERSION,
        "source": source,
        "encoder_config": _whisper_encoder_config(model),
        "state_dict_manifest": manifest,
        "state_dict_sha256": state_sha256,
        "state_dict": state_dict,
    }

    temporary_path = artifact_path.with_name(
        f".{artifact_path.name}.{os.getpid()}.tmp"
    )
    if temporary_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite temporary encoder artifact: {temporary_path}"
        )
    try:
        torch.save(payload, temporary_path)
        os.replace(temporary_path, artifact_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()

    artifact_sha256 = _file_sha256(artifact_path)
    metadata = {
        key: value for key, value in payload.items() if key != "state_dict"
    }
    metadata.update(
        {
            "operation": "export",
            "artifact_path": str(artifact_path),
            "artifact_sha256": artifact_sha256,
            "sidecar_path": str(sidecar_path),
            "tensor_count": len(state_dict),
            "parameter_numel": sum(
                int(tensor.numel()) for tensor in state_dict.values()
            ),
        }
    )
    _write_json(sidecar_path, metadata)
    return metadata


def _load_encoder_state_payload(path: Path) -> tuple[dict, str]:
    if not path.is_file():
        raise FileNotFoundError(f"Encoder state artifact does not exist: {path}")
    artifact_sha256 = _file_sha256(path)
    sidecar_path = _encoder_state_sidecar_path(path)
    if sidecar_path.exists():
        try:
            sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"Invalid encoder state sidecar: {sidecar_path}"
            ) from exc
        expected_artifact_sha256 = sidecar.get("artifact_sha256")
        if expected_artifact_sha256 != artifact_sha256:
            raise ValueError(
                "Encoder artifact SHA256 does not match its sidecar: "
                f"expected {expected_artifact_sha256!r}, got "
                f"{artifact_sha256!r}."
            )

    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ValueError(f"Could not safely load encoder state artifact: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError("Encoder state artifact payload must be a dictionary.")
    if payload.get("format") != _WHISPER_ENCODER_STATE_FORMAT:
        raise ValueError(
            f"Unsupported encoder state format: {payload.get('format')!r}."
        )
    if payload.get("format_version") != _WHISPER_ENCODER_STATE_VERSION:
        raise ValueError(
            "Unsupported encoder state format version: "
            f"{payload.get('format_version')!r}."
        )
    source = payload.get("source")
    if not isinstance(source, dict) or source.get("model_family") != "whisper":
        raise ValueError("Encoder state source must identify a Whisper model.")
    state_dict = payload.get("state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError("Encoder state artifact has no valid state_dict.")
    manifest = _state_dict_manifest(state_dict)
    if payload.get("state_dict_manifest") != manifest:
        raise ValueError("Encoder state_dict manifest does not match its tensors.")
    state_sha256 = _state_dict_sha256(state_dict)
    if payload.get("state_dict_sha256") != state_sha256:
        raise ValueError("Encoder state_dict SHA256 does not match its tensors.")
    return payload, artifact_sha256


def _import_whisper_encoder_state(
    model,
    path: str | Path,
    target_spec,
) -> dict:
    artifact_path = Path(path).expanduser().resolve()
    payload, artifact_sha256 = _load_encoder_state_payload(artifact_path)
    expected_config = _whisper_encoder_config(model)
    saved_config = payload.get("encoder_config")
    if saved_config != expected_config:
        raise ValueError(
            "Whisper encoder config mismatch between artifact and target: "
            f"saved={saved_config!r}, target={expected_config!r}."
        )

    encoder = _whisper_encoder(model)
    target_state = encoder.state_dict()
    imported_state = payload["state_dict"]
    expected_names = set(target_state)
    imported_names = set(imported_state)
    if expected_names != imported_names:
        raise ValueError(
            "Whisper encoder state_dict keys do not match target; "
            f"missing={sorted(expected_names - imported_names)}, "
            f"unexpected={sorted(imported_names - expected_names)}."
        )
    for name in sorted(expected_names):
        saved = imported_state[name]
        target = target_state[name]
        if saved.shape != target.shape:
            raise ValueError(
                f"Whisper encoder tensor {name!r} shape mismatch: "
                f"saved={list(saved.shape)}, target={list(target.shape)}."
            )
        if saved.dtype != target.dtype:
            raise ValueError(
                f"Whisper encoder tensor {name!r} dtype mismatch: "
                f"saved={saved.dtype}, target={target.dtype}."
            )

    encoder.load_state_dict(imported_state, strict=True)
    return {
        "operation": "import",
        "effective_model_state": "fp16-consumer-with-imported-encoder",
        "artifact_path": str(artifact_path),
        "artifact_sha256": artifact_sha256,
        "state_dict_sha256": payload["state_dict_sha256"],
        "format": payload["format"],
        "format_version": payload["format_version"],
        "source": payload["source"],
        "encoder_config": payload["encoder_config"],
        "state_dict_manifest": payload["state_dict_manifest"],
        "target": {
            "model_alias": target_spec.alias,
            "model_id": target_spec.model_id,
            "model_family": target_spec.family,
        },
    }


def _quantize_qwen(model, args) -> list[str]:
    if args.quant_scope not in {
        "full",
        "encoder",
        "text-backbone",
        "text-attention",
        "text-ffn",
    }:
        raise ValueError(
            "Qwen supports --quant-scope full, encoder (audio tower only), "
            "text-backbone, text-attention, or text-ffn."
        )
    calibration_data = _qwen_make_calibration_data(
        model,
        nsamples=args.nsamples,
        seed=args.seed,
        verbose=args.verbose,
        include_references=(
            args.score_calibration_wer and args.calibration_score_samples == 0
        ),
        include_labels=(
            args.sequence_hessian
            or args.sequence_calibration
            or getattr(args, "frame_weighting", "none") == "task-fisher"
        ),
    )
    if args.score_calibration_wer:
        args._calibration_data_for_scoring = calibration_data
    sequence_support = str(getattr(args, "sequence_support", "full"))
    if sequence_support != "full":
        if not (args.sequence_hessian or args.sequence_calibration):
            raise ValueError(
                "Non-full --sequence-support requires gptq+seqcal or "
                "gptq+seqhess so the full teacher-forced trajectory and "
                "prompt boundary are available."
            )
        support_seed = (
            int(args.sequence_support_seed)
            if int(args.sequence_support_seed) >= 0
            else int(args.quantization_seed)
        )
        args._qwen_sequence_support_masks = _qwen_sequence_support_masks(
            calibration_data,
            support=sequence_support,
            seed=support_seed,
        )
        prompt_counts = [
            int(sample["labels"].eq(-100).sum())
            for sample in calibration_data
        ]
        selected_response_counts = [
            int((mask & sample["labels"].ne(-100).cpu()).sum())
            for mask, sample in zip(
                args._qwen_sequence_support_masks,
                calibration_data,
            )
        ]
        metadata = getattr(args, "_quantization_metadata", {})
        metadata["sequence_support"] = {
            "mode": sequence_support,
            "score_trajectory": "complete teacher-forced transcript",
            "gptq_rows": (
                "prompt-prefix states only"
                if sequence_support == "prompt"
                else "fixed-count sample from the full trajectory"
            ),
            "support_seed": support_seed,
            "prompt_row_counts": prompt_counts,
            "selected_response_row_counts": selected_response_counts,
            "purpose": (
                "support-by-density identification arm"
                if sequence_support == "prompt"
                else "row-count-matched support identification arm"
            ),
        }
        args._quantization_metadata = metadata
    if getattr(args, "frame_weighting", "none") == "task-fisher":
        if not _qwen_scope_includes_stack(args.quant_scope, "audio"):
            raise ValueError(
                "Qwen task-fisher weights audio-tower rows and requires "
                "--quant-scope encoder or full."
            )
        audio_weights, summaries, mean_loss = (
            _qwen_collect_audio_task_fisher_weights(
                model,
                calibration_data,
                DEV,
                clip_max=float(args.propagated_clip_max),
                min_ess_fraction=float(args.task_fisher_min_ess_fraction),
            )
        )
        if getattr(args, "task_weight_permutation", "none") != "none":
            audio_weights = permute_calibration_token_weights(
                audio_weights,
                seed=(
                    int(args.task_weight_permutation_seed)
                    if int(args.task_weight_permutation_seed) >= 0
                    else int(args.quantization_seed)
                ),
            )
        args._qwen_audio_token_weights = audio_weights
        metadata = getattr(args, "_quantization_metadata", {})
        metadata["frame_weighting"] = {
            "mode": "task-fisher",
            "applied": True,
            "applies_to": "Qwen audio-tower linear groups",
            "estimand": (
                "g_t^(l) = mean_h |d L_TF / d h_t^(l)|^2; "
                "label-conditioned empirical-Fisher scalarization"
            ),
            "normalization": (
                "KL/I-projection per sample and layer: ratio-preserving "
                "final weights in "
                f"[{PROPAGATED_CLIP_MIN:g}, "
                f"{float(args.propagated_clip_max):g}] with unit mean"
            ),
            "mean_teacher_forced_loss": mean_loss,
            "minimum_ess_fraction": float(
                args.task_fisher_min_ess_fraction
            ),
            "layer_weight_summaries": summaries,
        }
        args._quantization_metadata = metadata
    if args.sequence_hessian:
        token_weights, summaries = _qwen_collect_sequence_token_weights(
            model,
            calibration_data,
            DEV,
            clip_max=float(args.propagated_clip_max),
        )
        permutation_mode = getattr(args, "task_weight_permutation", "none")
        permutation_seed = (
            int(args.task_weight_permutation_seed)
            if int(args.task_weight_permutation_seed) >= 0
            else int(args.quantization_seed)
        )
        if permutation_mode == "within-prompt-response":
            prompt_masks = _qwen_sequence_support_masks(
                calibration_data,
                support="prompt",
            )
            token_weights = permute_calibration_token_weights_stratified(
                token_weights,
                prompt_masks,
                seed=permutation_seed,
            )
        elif permutation_mode == "within-sample":
            token_weights = permute_calibration_token_weights(
                token_weights,
                seed=permutation_seed,
            )
        args._qwen_sequence_token_weights = token_weights
        metadata = getattr(args, "_quantization_metadata", {})
        metadata["sequence_hessian"] = {
            "objective": (
                "empirical_fisher_token_weighted_gptq_reconstruction"
            ),
            "hessian": "2 * X.T * diag(normalized_token_sensitivity) * X",
            "token_sensitivity": (
                "mean_squared_gradient_of_teacher_forced_sequence_loss_"
                "with_respect_to_text_mlp_down_proj_output"
            ),
            "normalization": "unit_mean_per_example_and_layer",
            "clipping": (
                "ratio-preserving KL/I-project to unit mean with final "
                "weights in [1e-4, "
                f"{float(args.propagated_clip_max):g}]"
            ),
            "teacher_forcing": "English transcript tokens; prompt labels masked",
            "layer_weight_summaries": summaries,
        }
        if permutation_mode != "none":
            metadata["sequence_hessian"]["task_weight_permutation"] = {
                "mode": permutation_mode,
                "seed": permutation_seed,
                "preserved": (
                    "prompt/response-stratum histogram, mass, bounds, and ESS"
                    if permutation_mode == "within-prompt-response"
                    else "per-sample/layer histogram, mass, bounds, and ESS"
                ),
            }
        args._quantization_metadata = metadata
    if args.method == "awq":
        _qwen_quantize_cached_sequential(
            model,
            calibration_data,
            args,
            DEV,
            alpha_by_module=None,
            collect_scores_only=False,
        )
        return [sample["__dataset_id__"] for sample in calibration_data]

    alpha_by_module = None
    if args.dynamic_alpha:
        score_model = copy.deepcopy(model)
        score_args = copy.deepcopy(args)
        score_args.qep = False
        module_scores = _qwen_quantize_cached_sequential(
            score_model,
            calibration_data,
            score_args,
            DEV,
            alpha_by_module=None,
            collect_scores_only=True,
        )
        del score_model
        torch.cuda.empty_cache()

        grouped_scores = {}
        for module_name, score in module_scores.items():
            key = _qwen_group_key_from_module_name(module_name)
            grouped_scores.setdefault(key, []).append(float(score))
        keys = sorted(grouped_scores)
        scores = [
            sum(grouped_scores[key]) / len(grouped_scores[key])
            for key in keys
        ]
        alphas = map_scores_to_alpha(
            scores,
            alpha_min=args.alpha_min,
            alpha_max=args.alpha_max,
            fallback_alpha=(args.alpha_min + args.alpha_max) / 2,
            log_scale=True,
            invert=False,
        )
        alpha_by_group = dict(zip(keys, map(float, alphas)))
        alpha_by_module = {
            name: alpha_by_group[_qwen_group_key_from_module_name(name)]
            for name in module_scores
        }
        metadata = getattr(args, "_quantization_metadata", {})
        metadata["dynamic_alpha"] = {
            "module_scores": {
                name: float(score) for name, score in module_scores.items()
            },
            "alpha_by_module": alpha_by_module,
        }
        args._quantization_metadata = metadata

    _qwen_quantize_cached_sequential(
        model,
        calibration_data,
        args,
        DEV,
        alpha_by_module=alpha_by_module,
        collect_scores_only=False,
    )
    return [sample["__dataset_id__"] for sample in calibration_data]


def _quantize_encoder_decoder_prepared(model, args, calibration_data) -> list[str]:
    reference_encoder = None
    if args.interface_bridge == "affine":
        reference_encoder = copy.deepcopy(model.model.encoder).cpu().eval()

    if args.dynamic_alpha:
        if args.quant_scope == "full":
            score_pack = quantize_with_dynamic_alpha(
                model,
                DEV,
                args,
                calibration_data,
                alpha_min=args.alpha_min,
                alpha_max=args.alpha_max,
            )
        elif args.quant_scope == "decoder":
            score_pack = quantize_decoder_with_dynamic_alpha(
                model,
                DEV,
                args,
                calibration_data,
                alpha_min=args.alpha_min,
                alpha_max=args.alpha_max,
            )
        else:
            raise ValueError(
                "Dynamic TIM-GPTQ legacy supports --quant-scope full or decoder "
                "for encoder-decoder models."
            )
        metadata = getattr(args, "_quantization_metadata", {})
        metadata["dynamic_alpha"] = {
            "layer_scores": score_pack["layer_scores"],
            "alpha_by_layer": score_pack["alpha_by_layer"],
        }
        args._quantization_metadata = metadata
    else:
        encoder_enabled = args.quant_scope in {
            "full",
            "encoder",
            "full-except-cross-attn",
        }
        decoder_enabled = args.quant_scope != "encoder"
        if encoder_enabled:
            encoder_result = quantize_encoder_with_alpha(
                model,
                DEV,
                args,
                calibration_data,
                collect_layer_mse=args.collect_context_scores,
            )
            if args.collect_context_scores:
                encoder_outputs, encoder_scores = encoder_result
                metadata = getattr(args, "_quantization_metadata", {})
                metadata["encoder_layer_scores"] = encoder_scores
                args._quantization_metadata = metadata
            else:
                encoder_outputs = encoder_result
        else:
            encoder_outputs = collect_fp_encoder_outputs(
                model,
                DEV,
                calibration_data,
            )
        if decoder_enabled:
            quantize_decoder_with_alpha(
                model,
                DEV,
                args,
                calibration_data,
                encoder_outputs,
            )

    if reference_encoder is not None:
        bridge_handle, bridge_metadata = fit_affine_encoder_bridge(
            model,
            reference_encoder,
            calibration_data,
            DEV,
        )
        args._interface_bridge_handle = bridge_handle
        args._quantization_metadata = {"interface_bridge": bridge_metadata}
        del reference_encoder
        torch.cuda.empty_cache()
    return [sample["dataset_id"] for sample in calibration_data]


def _decoder_ffn_output_modules(model) -> dict[str, torch.nn.Module]:
    modules = {}
    for index, layer in enumerate(model.model.decoder.layers):
        module = getattr(layer, "fc2", None)
        if not isinstance(module, torch.nn.Linear):
            raise TypeError(
                f"Decoder layer {index} does not expose a linear fc2 module."
            )
        modules[f"dec.layer{index}.fc2"] = module
    return modules


def _copy_module_weight(module, weight) -> None:
    module.weight.data.copy_(
        weight.to(device=module.weight.device, dtype=module.weight.dtype)
    )


def _remove_restored_module_from_quantized_allocation(args, module) -> None:
    allocation = getattr(args, "_bit_allocation_stats", None)
    if not allocation:
        return
    by_bits = allocation.get("weight_numel_by_bits", {})
    key = str(int(args.wbits))
    count = int(module.weight.numel())
    if int(by_bits.get(key, 0)) < count:
        raise ValueError(
            f"Cannot restore {count} parameters from W{key} allocation."
        )
    by_bits[key] = int(by_bits[key]) - count
    if by_bits[key] == 0:
        del by_bits[key]


def _quantize_with_sequence_protection(
    model,
    processor,
    args,
    calibration_data,
) -> list[str]:
    """Select one FP16 FFN output using downstream sequence-loss recovery."""
    selection_offset = int(args.nsamples)
    confirmation_offset = (
        selection_offset + int(args.sequence_selection_samples)
    )
    selection_data = get_asr_calibration_data(
        processor,
        nsamples=args.sequence_selection_samples,
        seed=args.seed,
        batch_size=args.calib_batch_size,
        offset=selection_offset,
    )
    confirmation_data = get_asr_calibration_data(
        processor,
        nsamples=args.sequence_confirmation_samples,
        seed=args.seed,
        batch_size=args.calib_batch_size,
        offset=confirmation_offset,
    )

    fp_selection = None
    fp_confirmation = None
    if args.sequence_objective == "fp-transcript":
        fp_selection = score_calibration_wer(
            model,
            processor,
            selection_data,
            DEV,
        )
        fp_confirmation = score_calibration_wer(
            model,
            processor,
            confirmation_data,
            DEV,
        )

    modules = _decoder_ffn_output_modules(model)
    fp_weights = {
        name: module.weight.detach().cpu().clone()
        for name, module in modules.items()
    }

    _quantize_encoder_decoder_prepared(model, args, calibration_data)
    quantized_weights = {
        name: module.weight.detach().cpu().clone()
        for name, module in modules.items()
    }

    if args.sequence_objective == "nll":
        score_candidate = lambda: score_calibration_nll(
            model,
            selection_data,
            DEV,
        )
        score_confirmation = lambda: score_calibration_nll(
            model,
            confirmation_data,
            DEV,
        )
        recovery_fn = paired_nll_recovery
        selection_metric = "mean_token_nll_recovery"
        objective_name = (
            "counterfactual_teacher_forced_mean_token_nll_recovery"
        )
    else:
        score_candidate = lambda: score_calibration_wer(
            model,
            processor,
            selection_data,
            DEV,
        )
        score_confirmation = lambda: score_calibration_wer(
            model,
            processor,
            confirmation_data,
            DEV,
        )
        recovery_fn = lambda plain, restored: paired_transcript_drift_recovery(
            fp_selection,
            plain,
            restored,
        )
        selection_metric = "sequence_drift_recovery"
        objective_name = (
            "counterfactual_autoregressive_fp16_transcript_drift_recovery"
        )

    plain_selection = score_candidate()
    candidate_scores = {}
    for name, module in modules.items():
        _copy_module_weight(module, fp_weights[name])
        restored_score = score_candidate()
        candidate_scores[name] = recovery_fn(
            plain_selection,
            restored_score,
        )
        _copy_module_weight(module, quantized_weights[name])

    selection = select_best_sequence_candidate(
        candidate_scores,
        metric=selection_metric,
    )
    plain_confirmation = score_confirmation()
    selected_name = selection["candidate"]
    selected_module = modules[selected_name]
    confirmation = None
    accepted = False
    if selection["eligible_for_confirmation"]:
        _copy_module_weight(selected_module, fp_weights[selected_name])
        restored_confirmation = score_confirmation()
        if args.sequence_objective == "nll":
            confirmation = paired_nll_recovery(
                plain_confirmation,
                restored_confirmation,
            )
            confirmation["bootstrap"] = paired_example_bootstrap_ci(
                confirmation["per_example"],
                confidence=args.sequence_confidence,
                replicates=args.sequence_bootstrap_replicates,
                seed=args.seed,
            )
        else:
            confirmation = paired_transcript_drift_recovery(
                fp_confirmation,
                plain_confirmation,
                restored_confirmation,
            )
            confirmation["bootstrap"] = paired_transcript_bootstrap_ci(
                confirmation["per_example"],
                confidence=args.sequence_confidence,
                replicates=args.sequence_bootstrap_replicates,
                seed=args.seed,
            )
        accepted = confirmation["bootstrap"]["lower"] > 0.0
        if not accepted:
            _copy_module_weight(
                selected_module,
                quantized_weights[selected_name],
            )

    if accepted:
        _remove_restored_module_from_quantized_allocation(args, selected_module)
        _record_fp16_protected_module(args, selected_name, selected_module)

    metadata = getattr(args, "_quantization_metadata", {})
    metadata["sequence_protection"] = {
        "objective": objective_name,
        "fit": {
            "offset": 0,
            "num_examples": len(calibration_data),
        },
        "selection": {
            "offset": selection_offset,
            "num_examples": len(selection_data),
            "candidate_scores": candidate_scores,
            **selection,
        },
        "confirmation": {
            "offset": confirmation_offset,
            "num_examples": len(confirmation_data),
            "candidate_score": confirmation,
        },
        "decision_rule": (
            "protect the selection-set winner only when its selection "
            "recovery is positive and the disjoint confirmation-set paired "
            "bootstrap confidence interval has lower bound above zero"
        ),
        "accepted": accepted,
        "protected_module": selected_name if accepted else None,
    }
    if args.sequence_objective == "nll":
        metadata["sequence_protection"]["selection"][
            "plain_mean_token_nll"
        ] = plain_selection["mean_token_nll"]
        metadata["sequence_protection"]["confirmation"][
            "plain_mean_token_nll"
        ] = plain_confirmation["mean_token_nll"]
    else:
        metadata["sequence_protection"]["selection"][
            "fp16_reference_wer"
        ] = fp_selection["wer"]
        metadata["sequence_protection"]["confirmation"][
            "fp16_reference_wer"
        ] = fp_confirmation["wer"]
    args._quantization_metadata = metadata
    return [sample["dataset_id"] for sample in calibration_data]


def _nll_summary(score):
    return {
        "num_examples": score["num_examples"],
        "num_tokens": score["num_tokens"],
        "mean_token_nll": score["mean_token_nll"],
    }


def _wer_summary(score):
    return {
        "num_examples": score["num_examples"],
        "wer": score["wer"],
    }


def _select_boundaryguard_candidate(
    *,
    fp_nll: float,
    uniform_nll: float,
    protected_nll: float,
    uniform_wer: float,
    protected_wer: float,
    min_wer_improvement: float,
    min_relative_nll_degradation: float,
):
    relative_nll_degradation = (uniform_nll - fp_nll) / max(abs(fp_nll), 1e-12)
    wer_improvement = uniform_wer - protected_wer
    wer_trigger = wer_improvement >= min_wer_improvement
    nll_trigger = (
        relative_nll_degradation >= min_relative_nll_degradation
        and protected_nll < uniform_nll
    )
    return {
        "selected": (
            "boundary-protected-w3" if wer_trigger or nll_trigger else "uniform-w3"
        ),
        "wer_trigger": wer_trigger,
        "nll_trigger": nll_trigger,
        "absolute_calibration_wer_improvement": wer_improvement,
        "relative_uniform_vs_fp16_nll_degradation": relative_nll_degradation,
    }


def _quantize_with_boundaryguard(model, processor, args, calibration_data) -> list[str]:
    """Select uniform or boundary-protected W3 from calibration-only signals."""
    fp_score = score_calibration_nll(model, calibration_data, DEV)
    fp_wer = score_calibration_wer(model, processor, calibration_data, DEV)
    model.cpu()
    torch.cuda.empty_cache()

    uniform_model = copy.deepcopy(model)
    uniform_args = copy.deepcopy(args)
    uniform_args.boundaryguard = False
    uniform_args.cross_attn_wbits = 0
    uniform_args.encoder_tail_wbits = 0
    uniform_args.encoder_tail_layers = 0
    uniform_args.encoder_promote_layers = ""
    uniform_args.encoder_promote_layer_indices = ()
    uniform_args.collect_context_scores = False
    uniform_args.score_calibration_nll = False
    _quantize_encoder_decoder_prepared(uniform_model, uniform_args, calibration_data)
    uniform_score = score_calibration_nll(uniform_model, calibration_data, DEV)
    uniform_wer = score_calibration_wer(
        uniform_model,
        processor,
        calibration_data,
        DEV,
    )
    uniform_model.cpu()
    torch.cuda.empty_cache()

    protected_args = copy.deepcopy(args)
    protected_args.boundaryguard = False
    protected_args.cross_attn_wbits = 4
    protected_args.encoder_tail_wbits = 4
    protected_args.encoder_tail_layers = 0
    last_layer = len(model.model.encoder.layers) - 1
    protected_args.encoder_promote_layers = f"0,{last_layer}"
    protected_args.encoder_promote_layer_indices = (0, last_layer)
    protected_args.collect_context_scores = False
    protected_args.score_calibration_nll = False
    _quantize_encoder_decoder_prepared(model, protected_args, calibration_data)
    protected_score = score_calibration_nll(model, calibration_data, DEV)
    protected_wer = score_calibration_wer(
        model,
        processor,
        calibration_data,
        DEV,
    )

    fp_nll = fp_score["mean_token_nll"]
    uniform_nll = uniform_score["mean_token_nll"]
    decision = _select_boundaryguard_candidate(
        fp_nll=fp_nll,
        uniform_nll=uniform_nll,
        protected_nll=protected_score["mean_token_nll"],
        uniform_wer=uniform_wer["wer"],
        protected_wer=protected_wer["wer"],
        min_wer_improvement=args.boundaryguard_min_calibration_wer_improvement,
        min_relative_nll_degradation=(
            args.boundaryguard_min_relative_nll_degradation
        ),
    )
    if decision["selected"] == "boundary-protected-w3":
        selected_args = protected_args
    else:
        selected_args = uniform_args
        model.cpu()
        model.load_state_dict(uniform_model.state_dict())

    args._bit_allocation_stats = selected_args._bit_allocation_stats
    args._quantization_metadata = {
        "boundaryguard": {
            "selection_metric": "dual-calibration-failure-detector",
            "fp16": {
                "nll": _nll_summary(fp_score),
                "greedy": _wer_summary(fp_wer),
            },
            "uniform": {
                "nll": _nll_summary(uniform_score),
                "greedy": _wer_summary(uniform_wer),
            },
            "boundary_protected": {
                "nll": _nll_summary(protected_score),
                "greedy": _wer_summary(protected_wer),
            },
            "selected": decision["selected"],
            "selection_triggers": {
                "greedy_wer": decision["wer_trigger"],
                "teacher_forced_nll": decision["nll_trigger"],
            },
            "protected_cross_attention_bits": 4,
            "protected_encoder_layers": [0, last_layer],
            "base_bits": 3,
            "relative_uniform_vs_fp16_nll_degradation": decision[
                "relative_uniform_vs_fp16_nll_degradation"
            ],
            "absolute_calibration_wer_improvement": decision[
                "absolute_calibration_wer_improvement"
            ],
            "minimum_absolute_calibration_wer_improvement": (
                args.boundaryguard_min_calibration_wer_improvement
            ),
            "minimum_relative_uniform_nll_degradation": (
                args.boundaryguard_min_relative_nll_degradation
            ),
        }
    }
    del uniform_model
    torch.cuda.empty_cache()
    return [sample["dataset_id"] for sample in calibration_data]


def _quantize_encoder_decoder(model, processor, args) -> list[str]:
    calibration_data = get_asr_calibration_data(
        processor,
        nsamples=args.nsamples,
        seed=args.seed,
        batch_size=args.calib_batch_size,
        include_feature_attention_mask=(
            # attention/propagated modes store the mask purely for
            # padding-vs-speech weight statistics; the weights themselves
            # come from the teacher-forced/propagated model passes.
            getattr(args, "frame_weighting", "none")
            in ("mask", "attention", "propagated", "task-fisher")
        ),
        augment=getattr(args, "calib_augment", "none"),
        augment_ratio=getattr(args, "calib_augment_ratio", 0.5),
    )
    if getattr(args, "calib_augment", "none") != "none":
        args._calibration_augmentation_records = [
            sample.get("augmentation") for sample in calibration_data
        ]
    _prepare_frame_weighting(model, args, calibration_data)
    if args.score_calibration_nll or args.score_calibration_wer:
        args._calibration_data_for_scoring = calibration_data
    if args.boundaryguard:
        return _quantize_with_boundaryguard(model, processor, args, calibration_data)
    if args.sequence_protect:
        return _quantize_with_sequence_protection(
            model,
            processor,
            args,
            calibration_data,
        )
    if args.sequence_hessian:
        token_weights, summaries = collect_decoder_sequence_token_weights(
            model,
            DEV,
            calibration_data,
        )
        args._sequence_token_weights = token_weights
        metadata = getattr(args, "_quantization_metadata", {})
        metadata["sequence_hessian"] = {
            "objective": (
                "empirical_fisher_token_weighted_gptq_reconstruction"
            ),
            "hessian": "2 * X.T * diag(normalized_token_sensitivity) * X",
            "token_sensitivity": (
                "mean_squared_gradient_of_teacher_forced_sequence_loss_"
                "with_respect_to_decoder_ffn_output"
            ),
            "normalization": "unit_mean_per_example_and_layer",
            "clipping": (
                "clip to [1e-4, 100] after initial normalization, "
                "then renormalize to unit mean"
            ),
            "layer_weight_summaries": summaries,
        }
        args._quantization_metadata = metadata
        return _quantize_encoder_decoder_prepared(
            model,
            args,
            calibration_data,
        )
    if args.propagation_hessian:
        token_weights, summaries = collect_decoder_propagation_token_weights(
            model,
            DEV,
            calibration_data,
            probes=args.propagation_hessian_probes,
        )
        args._sequence_token_weights = token_weights
        metadata = getattr(args, "_quantization_metadata", {})
        metadata["propagation_hessian"] = {
            "objective": (
                "one_block_lookahead_hidden_reconstruction_curvature"
            ),
            "hessian": "2 * X.T * diag(normalized_vjp_sensitivity) * X",
            "sensitivity_estimator": "Hutchinson diagonal of J.T @ J",
            "lookahead_blocks": 1,
            "rademacher_probes": int(args.propagation_hessian_probes),
            "target": (
                "next decoder block hidden state; final normalized decoder "
                "hidden state for the last block"
            ),
            "normalization": "unit_mean_per_example_and_layer",
            "clipping": (
                "clip to [1e-4, 100] after initial normalization, "
                "then renormalize to unit mean"
            ),
            "layer_weight_summaries": summaries,
        }
        args._quantization_metadata = metadata
        return _quantize_encoder_decoder_prepared(
            model,
            args,
            calibration_data,
        )
    return _quantize_encoder_decoder_prepared(model, args, calibration_data)


def _validate_frame_weighting_args(args) -> None:
    if getattr(args, "task_weight_permutation", "none") != "none":
        if not (
            getattr(args, "frame_weighting", "none") == "task-fisher"
            or bool(getattr(args, "sequence_hessian", False))
        ):
            raise ValueError(
                "--task-weight-permutation requires task-Fisher frame "
                "weighting or gptq+seqhess."
            )
    if getattr(args, "frame_weighting", "none") == "none":
        return
    if args.method != "gptq":
        raise ValueError(
            "--frame-weighting requires a GPTQ-derived mode; "
            f"mode {args.mode!r} uses method {args.method!r}."
        )
    if bool(getattr(args, "gptaq", False)) or bool(getattr(args, "qep", False)):
        raise ValueError(
            "--frame-weighting is not supported with dual-stream statistics "
            "modes (gptq+gptaq and QEP-derived modes): Helper.add_batch_qep "
            "does not accept token_weights. TODO: extend the dual-stream "
            "Gram accumulation with row weights before enabling this."
        )
    if args.frame_weighting in ("attention", "propagated"):
        floor = float(
            getattr(args, "frame_weighting_floor", DEFAULT_ATTENTION_FLOOR)
        )
        if not 0.0 <= floor < 1.0:
            raise ValueError(
                f"--frame-weighting-floor must be in [0, 1), got {floor}."
            )
    if args.frame_weighting in (
        "propagated",
        "propagated-uniform",
        "task-fisher",
    ):
        probes = int(
            getattr(args, "propagated_probes", DEFAULT_PROPAGATED_PROBES)
        )
        if probes <= 0:
            raise ValueError(
                f"--propagated-probes must be positive, got {probes}."
            )
        propagated_clip_max = float(
            getattr(args, "propagated_clip_max", PROPAGATED_CLIP_MAX)
        )
        if propagated_clip_max < PROPAGATED_CLIP_MIN:
            raise ValueError(
                "--propagated-clip-max must be at least "
                f"{PROPAGATED_CLIP_MIN}, got {propagated_clip_max}."
            )
    if args.frame_weighting == "task-fisher":
        min_ess_fraction = float(
            getattr(args, "task_fisher_min_ess_fraction", 0.0)
        )
        if not 0.0 <= min_ess_fraction <= 1.0:
            raise ValueError(
                "--task-fisher-min-ess-fraction must be in [0, 1], got "
                f"{min_ess_fraction}."
            )


def _validate_calibration_augment_args(args) -> None:
    if getattr(args, "calib_augment", "none") == "none":
        return
    ratio = float(getattr(args, "calib_augment_ratio", 0.5))
    if not 0.0 <= ratio <= 1.0:
        raise ValueError(
            f"--calib-augment-ratio must be in [0, 1], got {ratio}."
        )


def _validate_calibration_score_augment_args(args) -> None:
    mode = getattr(args, "calibration_score_augment", "none")
    ratio = float(getattr(args, "calibration_score_augment_ratio", 1.0))
    if not 0.0 <= ratio <= 1.0:
        raise ValueError(
            "--calibration-score-augment-ratio must be in [0, 1], "
            f"got {ratio}."
        )
    if mode != "none" and getattr(args, "calibration_score_samples", 0) <= 0:
        raise ValueError(
            "--calibration-score-augment requires "
            "--calibration-score-samples > 0."
        )


def _prepare_frame_weighting(model, args, calibration_data) -> None:
    """Compute per-sample encoder-frame weights and record run metadata.

    Runs before any weight is quantized: attention mode does its FP16
    teacher-forced pass directly on the live model (no copy).
    """
    if getattr(args, "frame_weighting", "none") == "none":
        return
    weights, info = compute_calibration_frame_weights(
        model,
        calibration_data,
        args.frame_weighting,
        floor=float(
            getattr(args, "frame_weighting_floor", DEFAULT_ATTENTION_FLOOR)
        ),
        probes=int(
            getattr(args, "propagated_probes", DEFAULT_PROPAGATED_PROBES)
        ),
        propagated_clip_max=float(
            getattr(args, "propagated_clip_max", PROPAGATED_CLIP_MAX)
        ),
        task_fisher_min_ess_fraction=float(
            getattr(args, "task_fisher_min_ess_fraction", 0.0)
        ),
    )
    if (
        args.frame_weighting == "task-fisher"
        and getattr(args, "task_weight_permutation", "none") != "none"
    ):
        permutation_seed = (
            int(args.task_weight_permutation_seed)
            if int(args.task_weight_permutation_seed) >= 0
            else int(args.quantization_seed)
        )
        weights = permute_calibration_token_weights(
            weights,
            seed=permutation_seed,
        )
        info["task_weight_permutation"] = {
            "mode": "within-sample",
            "seed": permutation_seed,
            "preserved": "per-sample/layer histogram, mean, bounds, ESS",
        }
    args._frame_token_weights = weights
    if args.frame_weighting in (
        "attention",
        "propagated",
        "propagated-uniform",
        "task-fisher",
    ):
        family_noop = (
            "moonshine input_values are variable-length and unpadded but "
            f"still receive {args.frame_weighting} weights; qwen3_asr uses "
            "its architecture-specific path for task-fisher only"
        )
    else:
        family_noop = (
            "moonshine input_values are variable-length and unpadded (weights "
            "None); qwen3_asr audio-tower weighting is task-fisher only"
        )
    metadata = getattr(args, "_quantization_metadata", {})
    metadata["frame_weighting"] = {
        **info,
        "row_weighting": (
            "Helper.add_batch token_weights: Gram rows scaled by sqrt(weight)"
        ),
        "applies_to": (
            "encoder linear groups and the decoder cross-attention K/V group "
            "(rows are encoder positions); decoder autoregressive token rows "
            "stay unweighted"
        ),
        "family_noop": family_noop,
    }
    args._quantization_metadata = metadata


def _validate_rotation_args(args) -> None:
    if getattr(args, "rotate", "none") == "none":
        return
    if args.method != "gptq":
        raise ValueError(
            "--rotate hadamard requires a GPTQ-derived mode; "
            f"mode {args.mode!r} uses method {args.method!r}."
        )
    if bool(getattr(args, "qep_gate", False)):
        raise ValueError(
            "--rotate hadamard is not supported with gptq+ffngate "
            "(the gate candidate scan bypasses the rotated GPTQ call)."
        )


def _record_new_baseline_metadata(args) -> None:
    """Record GPTAQ / rotation choices for run reproducibility."""
    if bool(getattr(args, "gptaq", False)):
        metadata = getattr(args, "_quantization_metadata", {})
        metadata["gptaq"] = {
            "reference": "arXiv:2504.02692 (GPTAQ/GPTQv2 asymmetric calibration)",
            "objective": (
                "match the quantized layer output on the quantized stream to "
                "the full-precision layer output on the full-precision stream"
            ),
            "alpha": float(args.gptaq_alpha),
            "correction": (
                "P = alpha * triu(dXXT Hinv^T, 1) Hinv added to the GPTQ "
                "column scan; dXXT = (X_fp - X_q) X_q^T"
            ),
        }
        args._quantization_metadata = metadata
    if getattr(args, "rotate", "none") != "none":
        metadata = getattr(args, "_quantization_metadata", {})
        metadata.setdefault(
            "rotation_config",
            {
                "rotate": str(args.rotate),
                "rotation_seed": int(args.rotation_seed),
            },
        )
        args._quantization_metadata = metadata


def quantize_model(model, processor, spec, args) -> list[str]:
    _validate_rotation_args(args)
    _validate_frame_weighting_args(args)
    _validate_calibration_augment_args(args)
    sequence_support = str(getattr(args, "sequence_support", "full"))
    permutation_mode = str(
        getattr(args, "task_weight_permutation", "none")
    )
    if sequence_support != "full" and spec.family != "qwen3_asr":
        raise ValueError("Sequence-support interventions are Qwen-specific.")
    if permutation_mode == "within-prompt-response":
        if spec.family != "qwen3_asr" or not args.sequence_hessian:
            raise ValueError(
                "Prompt/response-stratified permutation requires Qwen "
                "gptq+seqhess."
            )
        if sequence_support != "full":
            raise ValueError(
                "Prompt/response-stratified permutation requires full "
                "sequence support."
            )
        if getattr(args, "frame_weighting", "none") != "none":
            raise ValueError(
                "Prompt/response-stratified permutation applies only to "
                "sequence-Hessian weights, not audio frame weights."
            )
    if spec.family == "qwen3_asr":
        if getattr(args, "calib_augment", "none") != "none":
            raise ValueError(
                "--calib-augment is not wired into the Qwen calibration "
                "loader yet; use it with encoder-decoder models."
            )
        if getattr(args, "frame_weighting", "none") not in (
            "none",
            "task-fisher",
        ):
            metadata = getattr(args, "_quantization_metadata", {})
            metadata["frame_weighting"] = {
                "mode": args.frame_weighting,
                "applied": False,
                "reason": (
                    "qwen3_asr audio-tower capture path is out of scope; "
                    "frame weighting is a recorded no-op"
                ),
            }
            args._quantization_metadata = metadata
    _record_new_baseline_metadata(args)
    has_bit_override = (
        args.cross_attn_wbits > 0
        or args.encoder_tail_wbits > 0
        or args.encoder_tail_layers > 0
        or bool(args.encoder_promote_layer_indices)
    )
    if has_bit_override:
        if spec.family == "qwen3_asr":
            raise ValueError("Interface-aware bit overrides are not implemented for Qwen3-ASR.")
        if args.method != "gptq" or args.quant_scope != "full":
            raise ValueError(
                "Interface-aware bit overrides require a GPTQ-derived mode "
                "with --quant-scope full."
            )
        has_encoder_selection = (
            args.encoder_tail_layers > 0 or bool(args.encoder_promote_layer_indices)
        )
        if (args.encoder_tail_wbits > 0) != has_encoder_selection:
            raise ValueError(
                "--encoder-tail-wbits requires either --encoder-tail-layers or "
                "--encoder-promote-layers, and vice versa."
            )
        if args.encoder_tail_layers > 0 and args.encoder_promote_layer_indices:
            raise ValueError(
                "Use only one of --encoder-tail-layers and --encoder-promote-layers."
            )
    if args.interface_bridge != "none":
        if spec.family == "qwen3_asr":
            raise ValueError("Interface bridges are not implemented for Qwen3-ASR.")
        if args.method != "gptq" or args.quant_scope != "full":
            raise ValueError(
                "The affine interface bridge currently requires a GPTQ-derived "
                "mode with --quant-scope full."
            )
    if args.boundaryguard:
        if spec.family == "qwen3_asr":
            raise ValueError("BoundaryGuard currently targets encoder-decoder ASR models.")
        if args.wbits != 3 or args.quant_scope != "full":
            raise ValueError("BoundaryGuard requires W3 and --quant-scope full.")
        if has_bit_override or args.interface_bridge != "none":
            raise ValueError("BoundaryGuard manages its own bit allocation and bridge setting.")
    if args.sequence_protect:
        if spec.family == "qwen3_asr":
            raise ValueError(
                "Sequence protection currently targets encoder-decoder ASR models."
            )
        if args.method != "gptq" or args.quant_scope != "decoder":
            raise ValueError(
                "Sequence protection requires GPTQ with --quant-scope decoder."
            )
        if args.sequence_selection_samples <= 0:
            raise ValueError("--sequence-selection-samples must be positive.")
        if args.sequence_confirmation_samples <= 0:
            raise ValueError("--sequence-confirmation-samples must be positive.")
        if args.sequence_bootstrap_replicates <= 0:
            raise ValueError("--sequence-bootstrap-replicates must be positive.")
        if not 0.0 < args.sequence_confidence < 1.0:
            raise ValueError("--sequence-confidence must be in (0, 1).")
    if args.sequence_hessian:
        expected_scope = (
            "text-backbone" if spec.family == "qwen3_asr" else "decoder"
        )
        if args.method != "gptq" or args.quant_scope != expected_scope:
            raise ValueError(
                "Sequence-Hessian GPTQ requires GPTQ with "
                f"--quant-scope {expected_scope}."
            )
    if args.propagation_hessian:
        if spec.family == "qwen3_asr":
            raise ValueError(
                "Propagation-Hessian GPTQ is not yet implemented for Qwen."
            )
        if args.method != "gptq" or args.quant_scope != "decoder":
            raise ValueError(
                "Propagation-Hessian GPTQ requires GPTQ with "
                "--quant-scope decoder."
            )
        if args.propagation_hessian_probes <= 0:
            raise ValueError("--propagation-hessian-probes must be positive.")
    if args.sequence_calibration:
        if spec.family != "qwen3_asr":
            raise ValueError(
                "Sequence calibration is a Qwen-specific Hessian ablation."
            )
        if args.method != "gptq" or args.quant_scope != "text-backbone":
            raise ValueError(
                "Sequence calibration requires GPTQ with "
                "--quant-scope text-backbone."
            )
    if args.mode == "fp16":
        if args.score_calibration_wer or args.score_calibration_nll:
            if spec.family == "qwen3_asr":
                calibration_data = _qwen_make_calibration_data(
                    model,
                    nsamples=args.nsamples,
                    seed=args.seed,
                    verbose=args.verbose,
                    include_references=(
                        args.score_calibration_wer
                        and args.calibration_score_samples == 0
                    ),
                )
                args._calibration_data_for_scoring = calibration_data
                return [sample["__dataset_id__"] for sample in calibration_data]
            calibration_data = get_asr_calibration_data(
                processor,
                nsamples=args.nsamples,
                seed=args.seed,
                batch_size=args.calib_batch_size,
            )
            args._calibration_data_for_scoring = calibration_data
            return [sample["dataset_id"] for sample in calibration_data]
        return []
    if args.method == "rtn":
        rtn_targets = list(
            iter_rtn_named_modules(
                model,
                include_conv2d=False,
                include_lm_head=False,
                quant_scope=args.quant_scope,
            )
        )
        args._bit_allocation_stats = {
            "weight_numel_by_bits": {
                str(int(args.wbits)): sum(
                    int(module.weight.numel()) for _name, module in rtn_targets
                )
            }
        }
        rtn_quantize_model_inplace(
            model,
            wbits=args.wbits,
            group_size=args.groupsize,
            zero_point=True,
            include_conv2d=False,
            include_lm_head=False,
            skip_keywords=(),
            quant_scope=args.quant_scope,
            verbose=args.verbose,
        )
        return []
    if spec.family == "qwen3_asr":
        return _quantize_qwen(model, args)
    if not _supports_asr_pipeline(model):
        raise RuntimeError(
            f"Model '{spec.model_id}' does not expose the expected encoder/decoder layers."
        )
    return _quantize_encoder_decoder(model, processor, args)


def run(args) -> Path | None:
    if args.list_models:
        for spec in MODEL_SPECS.values():
            print(
                f"{spec.alias:18s} {spec.model_id:34s} "
                f"family={spec.family} groupsize={spec.group_size}"
            )
        return None

    spec = resolve_model_spec(args.model)
    args.model_alias = spec.alias
    args.model = spec.model_id
    args.model_family = spec.family
    args.groupsize = spec.group_size if args.groupsize == 0 else args.groupsize
    args.datasets = parse_datasets(args.datasets)
    resolve_mode(args)
    resolve_run_seeds(args)
    _validate_encoder_state_args(args, spec)
    if args.calibration_score_samples < 0 or args.calibration_score_offset < 0:
        raise ValueError("Calibration scoring sample count and offset must be nonnegative.")
    score_reuses_fit_distribution = (
        args.calibration_score_config == CALIBRATION_CONFIG
        and args.calibration_score_split == CALIBRATION_SPLIT
    )
    if (
        args.calibration_score_samples > 0
        and score_reuses_fit_distribution
        and args.calibration_score_offset < args.nsamples
    ):
        raise ValueError(
            "Held-out calibration scoring requires --calibration-score-offset "
            "to be at least --nsamples when scoring reuses the fit dataset."
        )
    _validate_calibration_score_augment_args(args)
    if (
        spec.family == "qwen3_asr"
        and (
            args.calibration_score_augment != "none"
            or not score_reuses_fit_distribution
        )
    ):
        raise ValueError(
            "Alternative scoring support and score augmentation are not wired "
            "into the Qwen calibration adapter."
        )

    resolved = vars(args).copy()
    resolved["protocol"] = {
        "calibration": {
            "dataset": CALIBRATION_DATASET_ID,
            "config": CALIBRATION_CONFIG,
            "split": CALIBRATION_SPLIT,
        },
        "calibration_scoring": {
            "dataset": CALIBRATION_DATASET_ID,
            "config": args.calibration_score_config,
            "split": args.calibration_score_split,
        },
        "evaluation_dataset": EVALUATION_DATASET_ID,
        "evaluation_splits": {
            alias: {"config": config, "split": split}
            for alias, (config, split) in DATASET_SPECS.items()
        },
    }
    if args.dry_run:
        print(json.dumps(resolved, indent=2, sort_keys=True))
        return None

    set_seed(args.quantization_seed)
    run_dir = _create_run_dir(args, spec.alias)
    _write_json(run_dir / "config.json", resolved)
    _write_json(run_dir / "environment.json", _environment_record())
    _write_json(run_dir / "status.json", {"state": "running"})

    started = time.perf_counter()
    try:
        model, processor = load_model(spec)
        model = _move_model_to_cuda_if_possible(model)
        model = _set_eval_mode(model)
        model_parameter_stats = _model_parameter_stats(model)
        encoder_state_metadata = None
        encoder_state_load_seconds = 0.0
        if args.load_encoder_state:
            load_started = time.perf_counter()
            encoder_state_metadata = _import_whisper_encoder_state(
                model,
                args.load_encoder_state,
                spec,
            )
            encoder_state_load_seconds = time.perf_counter() - load_started
            _write_json(run_dir / "encoder_state.json", encoder_state_metadata)

        quant_started = time.perf_counter()
        calibration_example_ids = quantize_model(model, processor, spec, args)
        quant_seconds = time.perf_counter() - quant_started
        if args.save_encoder_state:
            encoder_state_metadata = _export_whisper_encoder_state(
                model,
                args.save_encoder_state,
                spec,
                args,
            )
            _write_json(run_dir / "encoder_state.json", encoder_state_metadata)
        calibration_payload = {
            "num_examples": len(calibration_example_ids),
            "example_ids": calibration_example_ids,
        }
        augmentation_records = getattr(
            args, "_calibration_augmentation_records", None
        )
        if augmentation_records is not None:
            calibration_payload["augment"] = {
                "mode": args.calib_augment,
                "ratio": float(args.calib_augment_ratio),
                "records": augmentation_records,
            }
        _write_json(run_dir / "calibration.json", calibration_payload)
        quantization_metadata = getattr(args, "_quantization_metadata", {})
        if getattr(args, "frame_weighting", "none") != "none":
            frame_metadata = quantization_metadata.setdefault(
                "frame_weighting",
                {"mode": args.frame_weighting},
            )
            frame_metadata["per_module"] = finalize_frame_weight_statistics(args)
        allocation = getattr(args, "_bit_allocation_stats", None)
        if allocation:
            by_bits = allocation["weight_numel_by_bits"]
            total = sum(by_bits.values())
            effective_bits = (
                sum(int(bits) * count for bits, count in by_bits.items()) / total
            )
            quantization_metadata["bit_allocation"] = {
                "quantized_weight_numel_by_bits": by_bits,
                "quantized_weight_numel": total,
                "effective_bits_over_quantized_weights": effective_bits,
            }
            total_parameters = model_parameter_stats.get("total_parameter_numel", 0)
            if total_parameters >= total and total_parameters > 0:
                full_model_bits = (
                    effective_bits * total + 16.0 * (total_parameters - total)
                ) / total_parameters
                quantization_metadata["bit_allocation"].update(
                    {
                        **model_parameter_stats,
                        "quantized_parameter_fraction": total / total_parameters,
                        "effective_bits_over_all_parameters": full_model_bits,
                        "compression_ratio_vs_fp16_all_parameters": (
                            16.0 / full_model_bits
                        ),
                    }
                )
        if quantization_metadata:
            _write_json(run_dir / "quantization.json", quantization_metadata)
        scoring_data = None
        if args.score_calibration_nll or args.score_calibration_wer:
            if args.calibration_score_samples > 0:
                if spec.family == "qwen3_asr":
                    scoring_data = _qwen_make_calibration_data(
                        model,
                        nsamples=args.calibration_score_samples,
                        seed=args.seed,
                        verbose=args.verbose,
                        include_references=args.score_calibration_wer,
                        offset=args.calibration_score_offset,
                    )
                else:
                    scoring_data = get_asr_calibration_data(
                        processor,
                        nsamples=args.calibration_score_samples,
                        seed=args.seed,
                        batch_size=args.calib_batch_size,
                        offset=args.calibration_score_offset,
                        augment=args.calibration_score_augment,
                        augment_ratio=args.calibration_score_augment_ratio,
                        dataset_id=CALIBRATION_DATASET_ID,
                        config=args.calibration_score_config,
                        split=args.calibration_score_split,
                    )
            else:
                scoring_data = getattr(args, "_calibration_data_for_scoring")
        if args.score_calibration_nll:
            if spec.family == "qwen3_asr":
                raise ValueError("Calibration NLL scoring is not implemented for Qwen3-ASR.")
            calibration_score = score_calibration_nll(
                model,
                scoring_data,
                DEV,
            )
            _write_json(run_dir / "calibration_score.json", calibration_score)
        if args.score_calibration_wer:
            if spec.family == "qwen3_asr":
                calibration_wer = score_qwen_calibration_wer(
                    model,
                    scoring_data,
                    DEV,
                )
            else:
                calibration_wer = score_calibration_wer(
                    model,
                    processor,
                    scoring_data,
                    DEV,
                )
            calibration_wer["scoring_augmentation"] = {
                "mode": args.calibration_score_augment,
                "ratio": float(args.calibration_score_augment_ratio),
            }
            _write_json(run_dir / "calibration_wer.json", calibration_wer)

        metrics = {
            "quantization_seconds": quant_seconds,
            "effective_model_state": args.effective_model_state,
            "evaluations": [],
        }
        if args.load_encoder_state:
            metrics["encoder_state_load_seconds"] = encoder_state_load_seconds
            metrics["encoder_state"] = {
                "operation": "import",
                "artifact_path": encoder_state_metadata["artifact_path"],
                "artifact_sha256": encoder_state_metadata["artifact_sha256"],
                "state_dict_sha256": encoder_state_metadata["state_dict_sha256"],
                "source_model_alias": encoder_state_metadata["source"][
                    "model_alias"
                ],
            }
        elif args.save_encoder_state:
            metrics["encoder_state"] = {
                "operation": "export",
                "artifact_path": encoder_state_metadata["artifact_path"],
                "artifact_sha256": encoder_state_metadata["artifact_sha256"],
                "state_dict_sha256": encoder_state_metadata["state_dict_sha256"],
                "source_model_alias": encoder_state_metadata["source"][
                    "model_alias"
                ],
            }
        if args.eval:
            for dataset_name in args.datasets:
                result = evaluate_wer(
                    model,
                    processor,
                    DEV,
                    dataset_name=dataset_name,
                    max_samples=args.eval_samples,
                    batch_size=args.eval_batch_size,
                    manifest_dir=args.eval_manifest_dir or None,
                )
                predictions = result.pop("predictions")
                references = result.pop("references")
                example_ids = result.pop("example_ids")
                metrics["evaluations"].append(result)
                with (run_dir / f"{dataset_name}.jsonl").open(
                    "w", encoding="utf-8"
                ) as handle:
                    for index, (example_id, reference, prediction) in enumerate(
                        zip(example_ids, references, predictions)
                    ):
                        handle.write(
                            json.dumps(
                                {
                                    "index": index,
                                    "example_id": example_id,
                                    "reference": reference,
                                    "prediction": prediction,
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                _write_json(run_dir / "metrics.json", metrics)

        metrics["total_seconds"] = time.perf_counter() - started
        _write_json(run_dir / "metrics.json", metrics)
        _write_json(run_dir / "status.json", {"state": "completed"})
    except Exception as exc:
        _write_json(
            run_dir / "status.json",
            {
                "state": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "elapsed_seconds": time.perf_counter() - started,
            },
        )
        raise

    return run_dir


def main() -> None:
    args = build_parser().parse_args()
    run_dir = run(args)
    if run_dir is not None:
        print(f"Run artifacts: {run_dir}", flush=True)


if __name__ == "__main__":
    main()
