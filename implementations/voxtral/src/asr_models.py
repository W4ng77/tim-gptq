"""Model registry and loading for the supported ASR families."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from huggingface_hub import snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError
from transformers import (
    AutoProcessor,
    MoonshineForConditionalGeneration,
    WhisperForConditionalGeneration,
)


@dataclass(frozen=True)
class ModelSpec:
    alias: str
    model_id: str
    family: str
    group_size: int


MODEL_SPECS = {
    spec.alias: spec
    for spec in (
        ModelSpec("qwen-0.6b", "Qwen/Qwen3-ASR-0.6B", "qwen3_asr", 128),
        ModelSpec("qwen-1.7b", "Qwen/Qwen3-ASR-1.7B", "qwen3_asr", 128),
        ModelSpec("moonshine-tiny", "UsefulSensors/moonshine-tiny", "moonshine", 52),
        ModelSpec("moonshine-base", "UsefulSensors/moonshine-base", "moonshine", 72),
        ModelSpec("whisper-tiny", "openai/whisper-tiny", "whisper", 128),
        ModelSpec("whisper-base", "openai/whisper-base", "whisper", 128),
        ModelSpec("whisper-small", "openai/whisper-small", "whisper", 128),
        ModelSpec("whisper-medium", "openai/whisper-medium", "whisper", 128),
        ModelSpec("whisper-large-v3", "openai/whisper-large-v3", "whisper", 128),
        ModelSpec("voxtral-mini", "mistralai/Voxtral-Mini-3B-2507", "voxtral", 128),
    )
}


def _resolve_voxtral_source(model_id: str) -> str:
    """Prefer the local snapshot path over the hub repo id.

    mistral-common's tokenizer loader lists the hub repo when handed a repo
    id, which fails under HF_HUB_OFFLINE; a local directory path skips the
    listing entirely (recon-verified on transformers 4.57 + mistral-common
    1.9.1).
    """
    try:
        return snapshot_download(model_id, local_files_only=True)
    except LocalEntryNotFoundError:
        return model_id


def resolve_model_spec(model: str) -> ModelSpec:
    """Resolve a short alias or a supported Hugging Face model id."""
    if model in MODEL_SPECS:
        return MODEL_SPECS[model]

    for spec in MODEL_SPECS.values():
        if model == spec.model_id:
            return spec

    supported = ", ".join(sorted(MODEL_SPECS))
    raise ValueError(f"Unsupported model '{model}'. Supported aliases: {supported}")


def _ensure_qwen_pad_token_id(asr_wrapper) -> None:
    owners = []
    current = asr_wrapper
    for _ in range(4):
        if current is None:
            break
        owners.append(current)
        current = getattr(current, "model", None)

    eos_id = None
    for owner in owners:
        for config in (
            getattr(owner, "generation_config", None),
            getattr(owner, "config", None),
        ):
            value = getattr(config, "eos_token_id", None)
            if isinstance(value, (list, tuple)):
                value = value[0] if value else None
            if value is not None:
                eos_id = int(value)
                break
        if eos_id is not None:
            break

    if eos_id is None:
        return

    for owner in owners:
        for config in (
            getattr(owner, "generation_config", None),
            getattr(owner, "config", None),
        ):
            if config is not None and getattr(config, "pad_token_id", None) is None:
                config.pad_token_id = eos_id

        tokenizer = getattr(owner, "tokenizer", None)
        if tokenizer is not None and getattr(tokenizer, "pad_token_id", None) is None:
            tokenizer.pad_token_id = eos_id


def load_model(spec: ModelSpec):
    """Load one supported ASR model and its processor, if separate."""
    if spec.family == "qwen3_asr":
        from qwen_asr import Qwen3ASRModel

        model = Qwen3ASRModel.from_pretrained(
            spec.model_id,
            dtype=torch.bfloat16,
            device_map="cuda:0",
            max_inference_batch_size=32,
            max_new_tokens=256,
        )
        _ensure_qwen_pad_token_id(model)
        model.model.eval()
        return model, None

    if spec.family == "voxtral":
        from transformers import VoxtralForConditionalGeneration, VoxtralProcessor

        source = _resolve_voxtral_source(spec.model_id)
        processor = VoxtralProcessor.from_pretrained(source)
        model = VoxtralForConditionalGeneration.from_pretrained(
            source,
            dtype=torch.bfloat16,
        )
        model.eval()
        return model, processor

    processor = AutoProcessor.from_pretrained(spec.model_id)
    if spec.family == "moonshine":
        model = MoonshineForConditionalGeneration.from_pretrained(
            spec.model_id,
            dtype="auto",
        )
    elif spec.family == "whisper":
        model = WhisperForConditionalGeneration.from_pretrained(
            spec.model_id,
            dtype="auto",
        )
    else:
        raise ValueError(f"Unsupported model family: {spec.family}")

    model.eval()
    return model, processor
