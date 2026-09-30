"""Repair steps for damage that live phone recordings nearly always have:
rumble, clipped peaks and broadband noise."""

from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.ndimage import binary_dilation, uniform_filter
from scipy.signal import butter, sosfilt, sosfiltfilt


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


def find_ceilings(audio: np.ndarray) -> np.ndarray:
    """Per channel, the positive and negative extremes: shape ``(channels, 2)``."""
    return np.stack([audio.max(axis=-1), -audio.min(axis=-1)], axis=-1).astype(np.float64)


def plateau_counts(audio: np.ndarray, ceilings: np.ndarray, flat_tolerance: float = 0.002) -> np.ndarray:
    """Count flat-topped runs (3+ samples within ``flat_tolerance`` of the ceiling), per channel and polarity."""
    counts = np.zeros(ceilings.shape, dtype=np.int64)
    for ch in range(audio.shape[0]):
        for pol, sign in enumerate((1.0, -1.0)):
            ceiling = ceilings[ch, pol]
            if ceiling > 0.0:
                counts[ch, pol] = _count_runs(sign * audio[ch] >= (1.0 - flat_tolerance) * ceiling, 3)
    return counts


def clip_levels(ceilings: np.ndarray, counts: np.ndarray, min_plateaus: int = 3) -> np.ndarray:
    """Ceilings of the polarities that are actually clipped; NaN where they aren't."""
    return np.where((counts >= min_plateaus) & (ceilings > 0.0), ceilings, np.nan)


def declip(
    audio: np.ndarray,
    threshold: float = 0.98,
    flat_tolerance: float = 0.002,
    min_plateaus: int = 3,
    max_overshoot: float = 2.0,
    context: int = 8,
    levels: np.ndarray | None = None,
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

    For long recordings processed in blocks, pass ``levels`` from
    :func:`clip_levels` computed over the whole file so every block uses the
    same ceilings. Run this before any filtering, which would tilt the
    plateaus and hide them.

    Returns the repaired audio and the number of samples rebuilt.
    """
    out = np.array(audio, dtype=np.float32, copy=True)
    if levels is None:
        ceilings = find_ceilings(out)
        levels = clip_levels(ceilings, plateau_counts(out, ceilings, flat_tolerance), min_plateaus)
    repaired = 0
    for ch in range(out.shape[0]):
        x = out[ch]
        clipped = np.zeros(x.size, dtype=bool)
        for pol, sign in enumerate((1.0, -1.0)):
            level = levels[ch, pol]
            if np.isfinite(level):
                clipped |= _runs_at_least(sign * x >= threshold * level, 2)
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


class RumbleFilter:
    """Streaming version of :func:`remove_rumble`: filter state carries across blocks."""

    def __init__(self, sample_rate: int, cutoff_hz: float = 25.0, channels: int = 2):
        self._sos = butter(4, cutoff_hz, "highpass", fs=sample_rate, output="sos")
        self._zi = np.zeros((self._sos.shape[0], channels, 2))

    def process(self, block: np.ndarray) -> np.ndarray:
        # sosfilt wants zi shaped (sections, ..., 2) matching the non-filtered axes.
        out, self._zi = sosfilt(self._sos, block, axis=-1, zi=self._zi)
        return out.astype(np.float32)


def estimate_bandwidth(audio: np.ndarray, sample_rate: int, drop_db: float = 40.0) -> float:
    """Highest frequency with real content: where the spectrum falls ``drop_db`` under the midrange for good.

    Watch and voice-memo recordings are often cut off at 8-12 kHz; a normal
    phone recording reaches 16-20 kHz.
    """
    from scipy.signal import welch

    mono = np.asarray(audio, dtype=np.float64).mean(axis=0)
    if mono.size < 4096 or not np.any(mono):
        return sample_rate / 2.0
    freqs, psd = welch(mono, fs=sample_rate, nperseg=4096)
    db = 10.0 * np.log10(np.maximum(psd, 1e-20))
    reference = np.median(db[(freqs >= 1000) & (freqs <= 5000)])
    above = np.flatnonzero(db > reference - drop_db)
    return float(freqs[above[-1]]) if above.size else sample_rate / 2.0


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
    spec = stft(audio, n_fft, hop)
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

    return istft(spec * gain[None].astype(np.float32), n_fft, hop, n)


def _sqrt_hann(n_fft: int) -> np.ndarray:
    return np.sqrt(np.hanning(n_fft + 1)[:-1]).astype(np.float32)


def stft(audio: np.ndarray, n_fft: int = 2048, hop: int = 512) -> np.ndarray:
    """Short-time Fourier transform, ``(channels, bins, frames)``; pairs with :func:`istft`."""
    pad = n_fft // 2
    padded = np.pad(np.asarray(audio, dtype=np.float32), ((0, 0), (pad, pad + n_fft)))
    frames = np.lib.stride_tricks.sliding_window_view(padded, n_fft, axis=-1)[:, ::hop]
    frames = frames[:, : 1 + (padded.shape[-1] - n_fft) // hop]
    return np.fft.rfft(frames * _sqrt_hann(n_fft), axis=-1).astype(np.complex64).transpose(0, 2, 1)


def istft(spec: np.ndarray, n_fft: int, hop: int, length: int) -> np.ndarray:
    """Inverse of :func:`stft` by weighted overlap-add (exact reconstruction when unmodified)."""
    window = _sqrt_hann(n_fft)
    frames = np.fft.irfft(spec.transpose(0, 2, 1), n=n_fft, axis=-1).astype(np.float32) * window
    channels, count, _ = frames.shape
    ratio = n_fft // hop
    parts = frames.reshape(channels, count, ratio, hop)
    out = np.zeros((channels, count + ratio - 1, hop), dtype=np.float32)
    norm = np.zeros((count + ratio - 1, hop), dtype=np.float32)
    squared = (window**2).reshape(ratio, hop)
    for j in range(ratio):
        out[:, j : j + count] += parts[:, :, j]
        norm[j : j + count] += squared[j]
    out = out.reshape(channels, -1) / np.maximum(norm.reshape(-1), 1e-6)
    pad = n_fft // 2
    return np.ascontiguousarray(out[:, pad : pad + length], dtype=np.float32)


def restore_highs(audio: np.ndarray, sample_rate: int, cutoff_hz: float, amount: float = 0.5) -> np.ndarray:
    """Recreate the top octave a band-limited recording lost (a harmonic exciter).

    The octave just below the cutoff is gently saturated, which creates
    harmonics above it; only those new harmonics above the cutoff are added
    back, scaled to continue the recording's own spectral slope. It cannot
    recover the original cymbal detail, but restores air and brightness.
    """
    nyquist = sample_rate / 2.0
    if amount <= 0 or cutoff_hz >= nyquist * 0.9:
        return audio
    low_edge = cutoff_hz / 2.0
    sos = butter(4, [low_edge, cutoff_hz * 0.98], "bandpass", fs=sample_rate, output="sos")
    band = sosfiltfilt(sos, audio, axis=-1)
    band_rms = float(np.sqrt(np.mean(band**2))) or 1e-9
    drive = 3.0 / (band_rms * 10.0 + 1e-9)
    harmonics = np.tanh(band * drive) / drive + 0.5 * np.abs(band) - 0.5 * np.mean(np.abs(band))
    top = min(cutoff_hz * 2.0, nyquist * 0.95)
    sos_hi = butter(4, [cutoff_hz, top], "bandpass", fs=sample_rate, output="sos")
    new = sosfiltfilt(sos_hi, harmonics, axis=-1)
    new_rms = float(np.sqrt(np.mean(new**2))) or 1e-9
    # A falling spectrum: the new octave sits about 9 dB under the octave below it.
    target = band_rms * 10 ** (-9 / 20) * amount
    return (audio + new * (target / new_rms)).astype(np.float32)
