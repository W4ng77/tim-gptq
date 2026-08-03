"""Matched WER evaluation for Qwen3-ASR, Moonshine, and Whisper."""

from __future__ import annotations

import io
import json
import re
import string
from itertools import islice
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import tqdm
from jiwer import wer

from dataset_sources import load_local_parquet_dataset


EVALUATION_DATASET_ID = "hf-audio/open-asr-leaderboard"
# Voxtral transcription evaluation is pinned to English prompts (lang:en) and
# bounded generation; decoding stops at EOS well before this ceiling on the
# 30-second evaluation clips.
VOXTRAL_TRANSCRIPTION_LANGUAGE = "en"
VOXTRAL_EVAL_MAX_NEW_TOKENS = 440
DATASET_SPECS = {
    "librispeech-clean": ("librispeech", "test.clean"),
    "librispeech-other": ("librispeech", "test.other"),
    "spgispeech": ("spgispeech", "test"),
    "voxpopuli": ("voxpopuli", "test"),
    "gigaspeech": ("gigaspeech", "test"),
}


def normalize_asr_text(text: str) -> str:
    text = str(text).strip().lower()
    if "<asr_text>" in text:
        text = text.split("<asr_text>", 1)[1]
    text = re.sub(r"^language\s+\w+\s*", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return re.sub(r"\s+", " ", text).strip()


def _decode_audio(audio_info) -> np.ndarray:
    audio, sample_rate = sf.read(io.BytesIO(audio_info["bytes"]), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sample_rate != 16_000:
        target_length = int(round(len(audio) * 16_000 / sample_rate))
        audio = np.interp(
            np.linspace(0, len(audio) - 1, target_length),
            np.arange(len(audio)),
            audio,
        )
    return np.asarray(audio, dtype=np.float32)


def _expected_whisper_input_length(model) -> int:
    encoder = getattr(getattr(model, "model", None), "encoder", None)
    stride = 1
    if encoder is not None:
        for name in ("conv1", "conv2"):
            conv = getattr(encoder, name, None)
            if conv is not None:
                value = conv.stride[0] if isinstance(conv.stride, tuple) else conv.stride
                stride *= int(value)
    max_positions = getattr(model.config, "max_source_positions", None)
    return 3_000 if max_positions is None else int(max_positions) * max(stride, 1)


def _normalize_whisper_inputs(inputs: dict, model) -> dict:
    if "input_features" not in inputs:
        return inputs
    expected = _expected_whisper_input_length(model)
    features = inputs["input_features"]
    attention_mask = inputs.get("attention_mask")
    if features.shape[-1] < expected:
        padding = expected - features.shape[-1]
        features = F.pad(features, (0, padding))
        if attention_mask is not None:
            attention_mask = F.pad(attention_mask, (0, padding))
    elif features.shape[-1] > expected:
        features = features[..., :expected]
        if attention_mask is not None:
            attention_mask = attention_mask[..., :expected]
    inputs["input_features"] = features
    if attention_mask is not None:
        inputs["attention_mask"] = attention_mask
    return inputs


def _load_dataset(dataset_name: str, manifest_dir: str | None = None):
    try:
        config, split = DATASET_SPECS[dataset_name]
    except KeyError as exc:
        supported = ", ".join(sorted(DATASET_SPECS))
        raise ValueError(
            f"Unsupported evaluation dataset '{dataset_name}'. Supported: {supported}"
        ) from exc
    dataset = load_local_parquet_dataset(
        EVALUATION_DATASET_ID,
        config,
        split,
    )
    expected_ids = None
    manifest_path = None
    if manifest_dir:
        manifest_path = Path(manifest_dir).expanduser().resolve() / f"{dataset_name}.json"
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected = {
            "dataset_id": EVALUATION_DATASET_ID,
            "dataset_alias": dataset_name,
            "config": config,
            "split": split,
            "total_rows": dataset.num_rows,
        }
        mismatches = {
            key: (payload.get(key), value)
            for key, value in expected.items()
            if payload.get(key) != value
        }
        if mismatches:
            raise ValueError(f"Evaluation manifest mismatch in {manifest_path}: {mismatches}")
        selected = payload.get("selected", [])
        indices = [int(item["index"]) for item in selected]
        expected_ids = [str(item["id"]) for item in selected]
        dataset = dataset.select_indices(indices)
    return dataset, expected_ids, manifest_path


def _model_dtype(model):
    core = getattr(model, "model", model)
    return next(core.parameters()).dtype


def _is_voxtral_model(model) -> bool:
    config = getattr(model, "config", None)
    return getattr(config, "model_type", "") == "voxtral"


@torch.no_grad()
def _voxtral_transcribe_batch(model, processor, audios, device):
    """Transcribe decoded 16 kHz waveforms with a Voxtral transcription prompt.

    Each example is generated individually: the transcription-request prompt
    already ends in ``lang:en [TRANSCRIBE]``, and the generated continuation
    after the prompt length is the transcript.
    """
    model_id = str(
        getattr(getattr(model, "config", None), "_name_or_path", "")
        or "mistralai/Voxtral-Mini-3B-2507"
    )
    dtype = _model_dtype(model)
    predictions = []
    for audio in audios:
        inputs = processor.apply_transcription_request(
            language=VOXTRAL_TRANSCRIPTION_LANGUAGE,
            audio=audio,
            model_id=model_id,
            sampling_rate=16_000,
            format=["wav"],
        )
        inputs = {
            key: value.to(device)
            for key, value in inputs.items()
            if torch.is_tensor(value)
        }
        inputs["input_features"] = inputs["input_features"].to(dtype)
        generated = model.generate(
            **inputs,
            max_new_tokens=VOXTRAL_EVAL_MAX_NEW_TOKENS,
        )
        prompt_length = inputs["input_ids"].shape[1]
        predictions.append(
            processor.batch_decode(
                generated[:, prompt_length:],
                skip_special_tokens=True,
            )[0]
        )
    return predictions


def _move_model_to_device(model, device):
    if hasattr(model, "to"):
        try:
            moved = model.to(device)
            if moved is not None:
                model = moved
            return model
        except Exception:
            pass
    core = getattr(model, "model", None)
    if hasattr(core, "to"):
        core.to(device)
    return model


@torch.no_grad()
def score_calibration_nll(model, calibration_data, device) -> dict:
    """Compute teacher-forced token NLL on calibration transcripts."""
    model = _move_model_to_device(model, device)
    model.eval()
    total_nll = 0.0
    total_tokens = 0
    per_example = []
    dtype = _model_dtype(model)
    for sample in calibration_data:
        model_inputs = {}
        for key in ("input_features", "input_values"):
            if key in sample:
                model_inputs[key] = sample[key].to(device=device, dtype=dtype)
        labels = sample["decoder_input_ids"].to(device)
        output = model(**model_inputs, labels=labels)
        token_count = int(labels.numel())
        loss = float(output.loss.detach().float().item())
        total_nll += loss * token_count
        total_tokens += token_count
        per_example.append(
            {
                "example_id": sample["dataset_id"],
                "num_tokens": token_count,
                "mean_nll": loss,
            }
        )
    return {
        "num_examples": len(per_example),
        "num_tokens": total_tokens,
        "mean_token_nll": total_nll / max(total_tokens, 1),
        "per_example": per_example,
    }


@torch.no_grad()
def score_calibration_wer(model, processor, calibration_data, device) -> dict:
    """Run task-aligned greedy decoding on calibration examples."""
    model = _move_model_to_device(model, device)
    model.eval()
    dtype = _model_dtype(model)
    references = []
    predictions = []
    example_ids = []
    for sample in calibration_data:
        inputs = {}
        for key in ("input_features", "input_values"):
            if key in sample:
                inputs[key] = sample[key].to(device=device, dtype=dtype)
        model_type = getattr(model.config, "model_type", "")
        if model_type == "whisper":
            inputs = _normalize_whisper_inputs(inputs, model)
            generated = model.generate(
                **inputs,
                task="transcribe",
                language="en",
            )
        else:
            generated = model.generate(**inputs)
        prediction = processor.batch_decode(
            generated,
            skip_special_tokens=True,
        )[0]
        references.append(normalize_asr_text(sample["text"]))
        predictions.append(normalize_asr_text(prediction))
        example_ids.append(sample["dataset_id"])
    return {
        "num_examples": len(references),
        "wer": float(wer(references, predictions)),
        "example_ids": example_ids,
        "references": references,
        "predictions": predictions,
    }


@torch.no_grad()
def score_qwen_calibration_wer(model, calibration_data, device) -> dict:
    """Run Qwen transcription on calibration audio retained in memory."""
    rows = [
        sample
        for sample in calibration_data
        if "__audio__" in sample and "__text__" in sample
    ]
    if len(rows) != len(calibration_data):
        raise ValueError("Qwen calibration WER requires retained audio and references.")
    model = _move_model_to_device(model, device)
    results = model.transcribe(
        audio=[(sample["__audio__"], 16_000) for sample in rows],
        language=["English"] * len(rows),
        return_time_stamps=False,
    )
    predictions = [normalize_asr_text(result.text) for result in results]
    references = [normalize_asr_text(sample["__text__"]) for sample in rows]
    example_ids = [str(sample["__dataset_id__"]) for sample in rows]
    paired_count = min(len(predictions), len(references))
    predictions = predictions[:paired_count]
    references = references[:paired_count]
    example_ids = example_ids[:paired_count]
    return {
        "num_examples": len(references),
        "wer": float(wer(references, predictions)),
        "example_ids": example_ids,
        "references": references,
        "predictions": predictions,
    }


@torch.no_grad()
def score_voxtral_calibration_wer(model, processor, calibration_data, device) -> dict:
    """Run Voxtral transcription on held-out calibration waveforms."""
    rows = [
        sample
        for sample in calibration_data
        if "__audio__" in sample and "__text__" in sample
    ]
    if len(rows) != len(calibration_data):
        raise ValueError(
            "Voxtral calibration WER requires retained audio and references."
        )
    model = _move_model_to_device(model, device)
    predictions = [
        normalize_asr_text(text)
        for text in _voxtral_transcribe_batch(
            model,
            processor,
            [sample["__audio__"] for sample in rows],
            device,
        )
    ]
    references = [normalize_asr_text(sample["__text__"]) for sample in rows]
    example_ids = [str(sample["__dataset_id__"]) for sample in rows]
    paired_count = min(len(predictions), len(references))
    predictions = predictions[:paired_count]
    references = references[:paired_count]
    example_ids = example_ids[:paired_count]
    return {
        "num_examples": len(references),
        "wer": float(wer(references, predictions)),
        "example_ids": example_ids,
        "references": references,
        "predictions": predictions,
    }


@torch.no_grad()
def evaluate_wer(
    model,
    processor,
    device,
    dataset_name: str,
    max_samples: int = -1,
    batch_size: int = 8,
    manifest_dir: str | None = None,
) -> dict:
    """Evaluate one model on one named dataset and return raw run metadata."""
    dataset, expected_ids, manifest_path = _load_dataset(dataset_name, manifest_dir)
    batch_size = max(1, int(batch_size))
    is_qwen = processor is None and hasattr(model, "transcribe")
    predictions = []
    references = []
    example_ids = []

    model = _move_model_to_device(model, device)
    if hasattr(model, "eval"):
        model.eval()
    elif hasattr(model, "model"):
        model.model.eval()

    iterator = iter(dataset)
    remaining = None if max_samples == -1 else int(max_samples)
    progress = tqdm.tqdm(
        total=remaining,
        desc=dataset_name,
        unit="example",
    )
    try:
        while remaining is None or remaining > 0:
            take = batch_size if remaining is None else min(batch_size, remaining)
            rows = list(islice(iterator, take))
            if not rows:
                break

            audios = [_decode_audio(row["audio"]) for row in rows]
            batch_references = [normalize_asr_text(row["text"]) for row in rows]
            batch_ids = [str(row["id"]) for row in rows]
            if expected_ids is not None:
                start = len(example_ids)
                expected_batch_ids = expected_ids[start : start + len(batch_ids)]
                if batch_ids != expected_batch_ids:
                    raise RuntimeError(
                        f"Evaluation manifest ID check failed for {dataset_name}: "
                        f"expected {expected_batch_ids}, read {batch_ids}"
                    )

            if is_qwen:
                results = model.transcribe(
                    audio=[(audio, 16_000) for audio in audios],
                    language=["English"] * len(audios),
                    return_time_stamps=False,
                )
                batch_predictions = [
                    normalize_asr_text(result.text) for result in results
                ]
            elif _is_voxtral_model(model):
                batch_predictions = [
                    normalize_asr_text(text)
                    for text in _voxtral_transcribe_batch(
                        model,
                        processor,
                        audios,
                        device,
                    )
                ]
            else:
                inputs = processor(
                    audios,
                    sampling_rate=16_000,
                    return_tensors="pt",
                    padding=True,
                    return_attention_mask=True,
                )
                inputs = {
                    key: value.to(device)
                    for key, value in inputs.items()
                    if torch.is_tensor(value)
                }
                dtype = _model_dtype(model)
                for key in ("input_features", "input_values"):
                    if key in inputs:
                        inputs[key] = inputs[key].to(dtype)

                model_type = getattr(model.config, "model_type", "")
                if model_type == "whisper":
                    inputs = _normalize_whisper_inputs(inputs, model)
                    generated = model.generate(
                        **inputs,
                        task="transcribe",
                        language="en",
                    )
                else:
                    generated = model.generate(**inputs)
                batch_predictions = [
                    normalize_asr_text(text)
                    for text in processor.batch_decode(
                        generated,
                        skip_special_tokens=True,
                    )
                ]

            paired_count = min(len(batch_predictions), len(batch_references))
            predictions.extend(batch_predictions[:paired_count])
            references.extend(batch_references[:paired_count])
            example_ids.extend(batch_ids[:paired_count])
            progress.update(len(rows))
            if remaining is not None:
                remaining -= len(rows)
    finally:
        progress.close()
        close = getattr(iterator, "close", None)
        if close is not None:
            close()

    return {
        "dataset": dataset_name,
        "num_examples": len(references),
        "wer": float(wer(references, predictions)),
        "predictions": predictions,
        "references": references,
        "example_ids": example_ids,
        "manifest": None if manifest_path is None else str(manifest_path),
    }
