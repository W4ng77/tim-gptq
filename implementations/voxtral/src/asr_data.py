"""Deterministic calibration-data loading for encoder-decoder ASR models."""

from __future__ import annotations

import io
from itertools import islice

import numpy as np
import soundfile as sf
import tqdm

from calibration_augment import augment_waveform
from dataset_sources import load_local_parquet_dataset


TARGET_SAMPLE_RATE = 16_000
CALIBRATION_DATASET_ID = "openslr/librispeech_asr"
CALIBRATION_CONFIG = "clean"
CALIBRATION_SPLIT = "train.100"


def decode_audio(audio_info, target_sample_rate: int = TARGET_SAMPLE_RATE) -> np.ndarray:
    """Decode a datasets Audio(decode=False) record to mono float32."""
    audio, sample_rate = sf.read(io.BytesIO(audio_info["bytes"]), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    if sample_rate != target_sample_rate:
        target_length = int(round(len(audio) * target_sample_rate / sample_rate))
        audio = np.interp(
            np.linspace(0, len(audio) - 1, target_length),
            np.arange(len(audio)),
            audio,
        )
    return np.asarray(audio, dtype=np.float32)


def _tokenize_reference(processor, text: str):
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        raise TypeError("The processor does not expose a tokenizer for decoder calibration.")
    return tokenizer(
        text,
        return_tensors="pt",
        add_special_tokens=True,
    ).input_ids


def get_asr_calibration_data(
    processor,
    nsamples: int = 128,
    seed: int = 0,
    batch_size: int = 16,
    offset: int = 0,
    include_feature_attention_mask: bool = False,
    augment: str = "none",
    augment_ratio: float = 0.5,
):
    """Return deterministic LibriSpeech train-clean-100 calibration examples.

    ``batch_size`` is accepted for CLI compatibility. Feature extraction is
    intentionally per-example so variable-length Moonshine waveforms do not
    acquire padding-dependent calibration statistics.

    ``include_feature_attention_mask`` additionally stores the processor's
    frame-level attention mask as ``feature_attention_mask`` for models whose
    processor pads fixed-length ``input_features`` (Whisper: 3000 mel frames);
    variable-length ``input_values`` families (Moonshine) never get one.

    ``augment='acoustic'`` applies deterministic waveform-level augmentation
    (speed / gain / noise / reverb, chosen per ``(seed, offset + i)``) to
    roughly ``augment_ratio`` of the examples before feature extraction; each
    example records its augmentation parameters under ``augmentation``.
    """
    del batch_size
    if augment not in ("none", "acoustic"):
        raise ValueError(f"Unsupported calibration augmentation: {augment!r}")
    dataset = load_local_parquet_dataset(
        CALIBRATION_DATASET_ID,
        CALIBRATION_CONFIG,
        CALIBRATION_SPLIT,
    )
    dataset = dataset.shuffle(
        seed=seed,
        buffer_size=max(10_000, (int(offset) + int(nsamples)) * 20),
    )
    calibration_data = []

    iterator = iter(dataset)
    try:
        samples = islice(iterator, int(offset), int(offset) + int(nsamples))
        for position, sample in enumerate(
            tqdm.tqdm(
                samples,
                total=int(nsamples),
                desc="Preparing ASR calibration data",
            )
        ):
            audio = decode_audio(sample["audio"])
            augmentation_record = None
            if augment == "acoustic":
                audio, augmentation_record = augment_waveform(
                    audio,
                    seed=int(seed),
                    index=int(offset) + position,
                    ratio=float(augment_ratio),
                    sample_rate=TARGET_SAMPLE_RATE,
                )
            processor_kwargs = {}
            if include_feature_attention_mask:
                processor_kwargs["return_attention_mask"] = True
            processed = processor(
                audio,
                sampling_rate=TARGET_SAMPLE_RATE,
                return_tensors="pt",
                **processor_kwargs,
            )

            if "input_features" in processed:
                model_input = {"input_features": processed["input_features"]}
                if include_feature_attention_mask and "attention_mask" in processed:
                    model_input["feature_attention_mask"] = processed[
                        "attention_mask"
                    ]
            elif "input_values" in processed:
                model_input = {"input_values": processed["input_values"]}
            else:
                raise KeyError(
                    "Processor output contains neither input_features nor input_values: "
                    f"{list(processed.keys())}"
                )
            if augmentation_record is not None:
                model_input["augmentation"] = augmentation_record

            text = str(sample["text"])
            model_input["decoder_input_ids"] = _tokenize_reference(processor, text)
            model_input["text"] = text
            model_input["dataset_id"] = str(sample["id"])
            calibration_data.append(model_input)
    finally:
        close = getattr(iterator, "close", None)
        if close is not None:
            close()

    return calibration_data
