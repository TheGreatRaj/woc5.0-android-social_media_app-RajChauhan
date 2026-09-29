"""Repair steps for damage that live phone recordings nearly always have:
rumble, clipped peaks and broadband noise."""

from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.ndimage import binary_dilation, uniform_filter
from scipy.signal import butter, istft, sosfilt, sosfiltfilt, stft


def highpass(audio: np.ndarray, sample_rate: int, cutoff_hz: float, order: int = 2, zero_phase: bool = False) -> np.ndarray:
    sos = butter(order, cutoff_hz, "highpass", fs=sample_rate, output="sos")
    filt = sosfiltfilt if zero_phase else sosfilt
    return filt(sos, audio, axis=-1).astype(np.float32)


def lowpass(audio: np.ndarray, sample_rate: int, cutoff_hz: float, order: int = 2, zero_phase: bool = False) -> np.ndarray:
    sos = butter(order, cutoff_hz, "lowpass", fs=sample_rate, output="sos")
    filt = sosfiltfilt if zero_phase else sosfilt
    return filt(sos, audio, axis=-1).astype(np.float32)


def remove_rumble(audio: np.ndarray, sample_rate: int, cutoff_hz: float = 25.0) -> np.ndarray:
    """Strip DC offset, handling noise and wind thumps below ``cutoff_hz``."""
    audio = audio - audio.mean(axis=-1, keepdims=True)
    return highpass(audio, sample_rate, cutoff_hz, order=4)


def _runs_at_least(mask: np.ndarray, length: int) -> np.ndarray:
    """Keep only runs of True that are at least ``length`` samples long."""
    if length <= 1 or not mask.any():
        return mask
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    starts = np.flatnonzero(edges == 1)
    stops = np.flatnonzero(edges == -1)
    keep = np.zeros(mask.size + 1, dtype=np.int32)
    long_runs = (stops - starts) >= length
    np.add.at(keep, starts[long_runs], 1)
    np.add.at(keep, stops[long_runs], -1)
    return np.cumsum(keep[:-1]) > 0


def _count_runs(mask: np.ndarray, length: int) -> int:
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return int(np.sum((np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)) >= length))


def declip(
    audio: np.ndarray,
    threshold: float = 0.98,
    flat_tolerance: float = 0.002,
    min_plateaus: int = 3,
    max_overshoot: float = 2.0,
    context: int = 8,
) -> tuple[np.ndarray, int]:
    """Rebuild hard-clipped waveform peaks with cubic-spline interpolation.

    Phones and handheld recorders at loud gigs overload their input, leaving
    flat-topped plateaus at the ceiling. A polarity counts as clipped only if
    it has at least ``min_plateaus`` runs of 3+ samples within
    ``flat_tolerance`` of its extreme; natural waveform crests are curved and
    don't qualify. Samples in runs above ``threshold`` of the ceiling
    (plateau plus shoulders) are then re-estimated from ``context`` good
    samples either side. Rebuilt peaks never drop below the clipped value and
    are capped at ``max_overshoot`` times it (+6 dB), so a bad fit can't blow
    up. The output can exceed 1.0; later gain staging brings it back down.

    Run this on the decoded file before any filtering, which would tilt the
    plateaus and hide them.

    Returns the repaired audio and the number of samples rebuilt.
    """
    out = np.array(audio, dtype=np.float32, copy=True)
    repaired = 0
    for ch in range(out.shape[0]):
        x = out[ch]
        clipped = np.zeros(x.size, dtype=bool)
        for polarity in (1.0, -1.0):
            signed = polarity * x
            ceiling = float(signed.max())
            if ceiling <= 0.0:
                continue
            if _count_runs(signed >= (1.0 - flat_tolerance) * ceiling, 3) < min_plateaus:
                continue
            clipped |= _runs_at_least(signed >= threshold * ceiling, 2)
        if not clipped.any():
            continue
        # Only good samples near a clipped run are needed to fit the spline.
        known = np.flatnonzero(binary_dilation(clipped, iterations=context) & ~clipped)
        if known.size < 4:
            continue
        missing = np.flatnonzero(clipped)
        floor = np.abs(x[missing])
        estimate = np.clip(np.abs(CubicSpline(known, x[known])(missing)), floor, max_overshoot * floor)
        x[missing] = np.sign(x[missing]) * estimate
        repaired += missing.size
    return out, repaired


def spectral_denoise(
    audio: np.ndarray,
    sample_rate: int,
    strength: float = 0.5,
    max_reduction_db: float = 15.0,
    noise_percentile: float = 10.0,
    n_fft: int = 2048,
) -> np.ndarray:
    """Stationary noise reduction by spectral subtraction.

    The noise floor of each frequency bin is estimated as a low percentile of
    its power over time, which suits sparse material such as an isolated vocal
    where gaps between phrases expose the hiss. Gains are smoothed across time
    and frequency and floored at ``-max_reduction_db`` to avoid the warbling
    "musical noise" of hard spectral gating.
    """
    if strength <= 0.0:
        return audio
    n = audio.shape[-1]
    hop = n_fft // 4
    _, _, spec = stft(audio, fs=sample_rate, nperseg=n_fft, noverlap=n_fft - hop, axis=-1)
    power = np.mean(np.abs(spec) ** 2, axis=0)
    frame_energy = power.sum(axis=0)
    active = frame_energy > frame_energy.max() * 1e-8
    if active.sum() < 8:
        return audio
    # Noise power in an STFT bin is exponentially distributed, so a low
    # percentile sits far below the mean; scale it back up to the mean.
    q = noise_percentile / 100.0
    noise = np.percentile(power[:, active], noise_percentile, axis=1)[:, None] / -np.log1p(-q)

    over_subtraction = 1.0 + 3.0 * strength
    gain = np.sqrt(np.maximum(1.0 - over_subtraction * noise / np.maximum(power, 1e-20), 0.0))
    gain = uniform_filter(gain, size=(3, 5), mode="nearest")
    floor = 10.0 ** (-max_reduction_db * min(1.0, strength * 2.0) / 20.0)
    gain = np.maximum(gain, floor)

    _, cleaned = istft(spec * gain[None], fs=sample_rate, nperseg=n_fft, noverlap=n_fft - hop)
    cleaned = cleaned[..., :n]
    if cleaned.shape[-1] < n:
        cleaned = np.pad(cleaned, ((0, 0), (0, n - cleaned.shape[-1])))
    return cleaned.astype(np.float32)
