"""Deterministic acoustic augmentation for ASR calibration waveforms.

Applied to the mono float32 waveform after ``decode_audio`` and before the
processor, so it is model-family agnostic. Each calibration example draws its
augmentation deterministically from ``(seed, index)`` via
``numpy.random.default_rng([seed, index])``; roughly ``ratio`` of the samples
are augmented and the rest pass through unchanged.

Augmentation types (uniformly chosen among the available ones):

- speed:  resampling-based speed perturbation, factor 0.9 or 1.1.
- gain:   +6 dB or -6 dB.
- noise:  additive white Gaussian noise at an SNR drawn from [5, 20] dB.
- reverb: convolution with a synthetic exponentially-decaying Gaussian RIR
  (direct path + tail, scipy.signal.fftconvolve). If scipy is unavailable
  the sample degrades to a recorded skip and passes through unchanged.

Every sample returns a JSON-serializable record of what was (not) applied,
which the experiment CLI persists in ``calibration.json``.
"""

from __future__ import annotations

import numpy as np

try:  # Optional dependency: reverb degrades to a recorded skip without it.
    from scipy.signal import fftconvolve

    _HAVE_SCIPY = True
except ImportError:  # pragma: no cover - exercised only without scipy
    fftconvolve = None
    _HAVE_SCIPY = False


AUGMENT_CHOICES = ("none", "acoustic")
AUGMENT_TYPES = ("speed", "gain", "noise", "reverb")

SPEED_FACTORS = (0.9, 1.1)
GAIN_DB_CHOICES = (-6.0, 6.0)
NOISE_SNR_DB_RANGE = (5.0, 20.0)
REVERB_TAIL_SECONDS = 0.3
REVERB_DECAY_RANGE_SECONDS = (0.05, 0.15)


def _resample_speed(audio: np.ndarray, factor: float) -> np.ndarray:
    """Change playback speed by ``factor`` via linear-interpolation resampling."""
    target_length = max(int(round(len(audio) / float(factor))), 2)
    positions = np.linspace(0.0, len(audio) - 1.0, target_length)
    return np.interp(positions, np.arange(len(audio)), audio).astype(np.float32)


def _apply_gain(audio: np.ndarray, gain_db: float) -> np.ndarray:
    return (audio * float(10.0 ** (gain_db / 20.0))).astype(np.float32)


def _add_noise(audio: np.ndarray, snr_db: float, rng: np.random.Generator):
    signal_power = float(np.mean(np.square(audio)))
    if signal_power <= 0.0:
        return audio, {"skipped": True, "reason": "silent-signal"}
    noise = rng.standard_normal(len(audio)).astype(np.float32)
    noise_power = float(np.mean(np.square(noise)))
    scale = float(np.sqrt(signal_power / (noise_power * 10.0 ** (snr_db / 10.0))))
    return (audio + scale * noise).astype(np.float32), {"noise_scale": scale}


def _apply_reverb(
    audio: np.ndarray,
    decay_seconds: float,
    rng: np.random.Generator,
    sample_rate: int,
):
    if not _HAVE_SCIPY:
        return audio, {"skipped": True, "reason": "scipy-not-available"}
    tail_length = max(int(REVERB_TAIL_SECONDS * sample_rate), 2)
    times = np.arange(tail_length, dtype=np.float32) / float(sample_rate)
    rir = rng.standard_normal(tail_length).astype(np.float32)
    rir *= np.exp(-times / float(decay_seconds)).astype(np.float32)
    rir[0] = 1.0  # direct path
    rir /= float(np.sqrt(np.sum(np.square(rir)))) or 1.0
    wet = fftconvolve(audio, rir)[: len(audio)].astype(np.float32)
    input_rms = float(np.sqrt(np.mean(np.square(audio))))
    wet_rms = float(np.sqrt(np.mean(np.square(wet))))
    if wet_rms > 0.0 and input_rms > 0.0:
        wet = (wet * (input_rms / wet_rms)).astype(np.float32)
    return wet, {"rir_tail_seconds": REVERB_TAIL_SECONDS}


def augment_waveform(
    audio: np.ndarray,
    seed: int,
    index: int,
    ratio: float = 0.5,
    sample_rate: int = 16_000,
):
    """Deterministically augment one calibration waveform.

    Returns ``(waveform, record)``. The same ``(seed, index)`` always yields
    the same waveform and record; ``ratio`` is the probability that this
    sample is augmented at all.
    """
    if not 0.0 <= float(ratio) <= 1.0:
        raise ValueError(f"Augmentation ratio must be in [0, 1], got {ratio}.")
    rng = np.random.default_rng([int(seed), int(index)])
    record = {"index": int(index), "seed": int(seed), "applied": False}
    if float(rng.random()) >= float(ratio):
        return audio, record

    kind = str(rng.choice(AUGMENT_TYPES))
    record.update({"applied": True, "type": kind})
    if kind == "speed":
        factor = float(rng.choice(SPEED_FACTORS))
        record["speed_factor"] = factor
        return _resample_speed(audio, factor), record
    if kind == "gain":
        gain_db = float(rng.choice(GAIN_DB_CHOICES))
        record["gain_db"] = gain_db
        return _apply_gain(audio, gain_db), record
    if kind == "noise":
        snr_db = float(rng.uniform(*NOISE_SNR_DB_RANGE))
        record["snr_db"] = snr_db
        augmented, extra = _add_noise(audio, snr_db, rng)
        record.update(extra)
        if extra.get("skipped"):
            record["applied"] = False
        return augmented, record
    if kind == "reverb":
        decay = float(rng.uniform(*REVERB_DECAY_RANGE_SECONDS))
        record["decay_seconds"] = decay
        augmented, extra = _apply_reverb(audio, decay, rng, sample_rate)
        record.update(extra)
        if extra.get("skipped"):
            record["applied"] = False
        return augmented, record
    raise RuntimeError(f"Unreachable augmentation type: {kind!r}")
