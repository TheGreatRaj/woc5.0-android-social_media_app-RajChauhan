"""Level, loudness and peak measurements shared by the processing stages."""

from __future__ import annotations

import warnings

import numpy as np
import pyloudnorm
from scipy.ndimage import uniform_filter1d
from scipy.signal import resample_poly

SILENCE_DB = -120.0


def db_to_gain(db: float | np.ndarray) -> float | np.ndarray:
    return 10.0 ** (np.asarray(db) / 20.0)


def gain_to_db(gain: float | np.ndarray) -> float | np.ndarray:
    return 20.0 * np.log10(np.maximum(np.abs(gain), 1e-12))


def frame_rms_db(audio: np.ndarray, sample_rate: int, frame_ms: float = 50.0) -> np.ndarray:
    """RMS level of consecutive frames in dBFS, linked across channels."""
    frame = max(1, int(sample_rate * frame_ms / 1000))
    power = np.mean(np.square(audio, dtype=np.float64), axis=0)
    n_frames = power.size // frame
    if n_frames == 0:
        frames = power.mean(keepdims=True)
    else:
        frames = power[: n_frames * frame].reshape(n_frames, frame).mean(axis=1)
    return 10.0 * np.log10(np.maximum(frames, 1e-12))


def active_level_db(audio: np.ndarray, sample_rate: int, percentile: float = 95.0) -> float | None:
    """Level of the loud passages of a signal, ignoring near-silence.

    Thresholds for gates and compressors are set relative to this, so a stem
    gets the same treatment whether it was separated hot or quiet. Returns
    ``None`` for effectively silent audio.
    """
    levels = frame_rms_db(audio, sample_rate)
    active = levels[levels > -70.0]
    if active.size == 0:
        return None
    return float(np.percentile(active, percentile))


def integrated_lufs(audio: np.ndarray, sample_rate: int) -> float:
    """ITU-R BS.1770 integrated loudness; ``-inf`` for silence."""
    min_len = int(0.4 * sample_rate) + 1
    if audio.shape[-1] < min_len:
        audio = np.pad(audio, ((0, 0), (0, min_len - audio.shape[-1])))
    meter = pyloudnorm.Meter(sample_rate)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(meter.integrated_loudness(np.asarray(audio, dtype=np.float64).T))


def true_peak_envelope(audio: np.ndarray, oversample: int = 4, chunk: int = 1 << 18) -> np.ndarray:
    """Per-sample inter-sample peak magnitude, linked across channels.

    Uses 4x polyphase oversampling as in BS.1770 and works in chunks so long
    recordings don't need four times their size in memory.
    """
    n = audio.shape[-1]
    pad = 64
    out = np.empty(n, dtype=np.float32)
    for start in range(0, n, chunk):
        stop = min(n, start + chunk)
        lo, hi = max(0, start - pad), min(n, stop + pad)
        up = resample_poly(audio[:, lo:hi], oversample, 1, axis=-1)
        up = np.abs(up).max(axis=0)
        seg = up[(start - lo) * oversample : (stop - lo) * oversample]
        seg = seg[: (stop - start) * oversample].reshape(stop - start, oversample).max(axis=1)
        out[start:stop] = np.maximum(seg, np.abs(audio[:, start:stop]).max(axis=0))
    return out


def true_peak_db(audio: np.ndarray) -> float:
    if audio.size == 0:
        return SILENCE_DB
    return float(gain_to_db(true_peak_envelope(audio).max()))


def smoothed_envelope_db(audio: np.ndarray, sample_rate: int, window_ms: float) -> np.ndarray:
    """Short-window RMS envelope in dB, linked across channels."""
    window = max(1, int(sample_rate * window_ms / 1000))
    power = uniform_filter1d(np.mean(np.square(audio, dtype=np.float64), axis=0), window, mode="nearest")
    return 10.0 * np.log10(np.maximum(power, 1e-12))


def is_mono(audio: np.ndarray, threshold_db: float = -35.0) -> bool:
    """True when the side channel is negligible compared to the mid channel."""
    if audio.shape[0] < 2:
        return True
    mid = 0.5 * (audio[0] + audio[1])
    side = 0.5 * (audio[0] - audio[1])
    mid_power = float(np.mean(np.square(mid, dtype=np.float64)))
    side_power = float(np.mean(np.square(side, dtype=np.float64)))
    if mid_power <= 0.0:
        return side_power <= 0.0
    return 10.0 * np.log10(max(side_power, 1e-20) / mid_power) < threshold_db
