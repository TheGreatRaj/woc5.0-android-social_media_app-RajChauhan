"""Mastering: bus glue, tonal balance, loudness and a true-peak-safe limiter."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from pedalboard import Compressor, Pedalboard
from scipy.ndimage import minimum_filter1d, uniform_filter1d
from scipy.signal import fftconvolve, firwin2, welch

from .analysis import active_level_db, db_to_gain, gain_to_db, integrated_lufs, true_peak_db, true_peak_envelope

# Long-term average spectra of commercial mixes fall roughly 2-4.5 dB per octave
# in third-octave band energy between 80 Hz and 12 kHz, depending on genre.
# Phone recordings of gigs usually measure flatter (brighter and thinner).
RELEASE_TILT_DB_PER_OCTAVE = -3.0


@dataclass
class MasterReport:
    input_lufs: float
    output_lufs: float
    true_peak_dbtp: float
    loudness_gain_db: float
    max_limiting_db: float
    eq_bands_hz: list[float] = field(default_factory=list)
    eq_curve_db: list[float] = field(default_factory=list)


def glue_compress(audio: np.ndarray, sample_rate: int, depth_db: float = 6.0) -> np.ndarray:
    """Slow 2:1 bus compression so the separately processed stems gel again."""
    loud_db = active_level_db(audio, sample_rate)
    if loud_db is None:
        return audio
    board = Pedalboard([Compressor(threshold_db=loud_db - depth_db, ratio=2.0, attack_ms=30.0, release_ms=200.0)])
    return board(np.ascontiguousarray(audio, dtype=np.float32), sample_rate)


def band_spectrum_db(
    audio: np.ndarray, sample_rate: int, f_min: float = 31.5, f_max: float = 16000.0, bands_per_octave: int = 3
) -> tuple[np.ndarray, np.ndarray]:
    """Long-term energy per fractional-octave band, in dB."""
    mono = np.asarray(audio, dtype=np.float64).mean(axis=0)
    nperseg = min(8192, mono.size)
    freqs, psd = welch(mono, fs=sample_rate, nperseg=nperseg)
    n_bands = int(np.floor(np.log2(f_max / f_min) * bands_per_octave)) + 1
    centers = f_min * 2.0 ** (np.arange(n_bands) / bands_per_octave)
    half = 2.0 ** (1.0 / (2 * bands_per_octave))
    df = freqs[1] - freqs[0]
    energy = np.array([psd[(freqs >= c / half) & (freqs < c * half)].sum() * df for c in centers])
    return centers, 10.0 * np.log10(np.maximum(energy, 1e-20))


def tonal_correction_db(
    audio: np.ndarray,
    sample_rate: int,
    reference: np.ndarray | None = None,
    strength: float = 0.5,
    max_db: float = 4.0,
) -> tuple[np.ndarray, np.ndarray]:
    """EQ curve (band centres, gains in dB) that moves ``audio`` toward a target.

    With a reference track the target is its spectrum (a "sounds like the
    record" match). Without one, the overall tilt is pulled toward a typical
    release slope and narrow resonant peaks, usually room modes of the venue,
    are cut. Dips are never filled: they are usually cancellations that EQ
    can't fix.
    """
    centers, current = band_spectrum_db(audio, sample_rate)
    fit = (centers >= 80.0) & (centers <= 12000.0)
    octaves = np.log2(centers / 1000.0)

    if reference is not None:
        _, target = band_spectrum_db(reference, sample_rate)
        diff = target - current
        # Level is set later by loudness normalisation; centre the curve so
        # only the shape is matched.
        diff -= np.median(diff[fit])
    else:
        slope, _ = np.polyfit(octaves[fit], current[fit], 1)
        tilt = (RELEASE_TILT_DB_PER_OCTAVE - slope) * octaves
        # Anything sticking out of the one-octave-smoothed spectrum is a resonance.
        smooth = np.convolve(np.pad(current, 1, mode="edge"), np.ones(3) / 3.0, mode="valid")
        resonance = -np.maximum(current - smooth, 0.0)
        diff = tilt + resonance

    curve = np.clip(diff * strength, -max_db, max_db)
    # Don't push into sub-bass or the extreme top: phone mics recorded nothing there.
    curve = np.where(centers < 45.0, np.minimum(curve, 0.0), curve)
    curve = np.where(centers > 14000.0, np.minimum(curve, 1.0), curve)
    return centers, curve


def apply_eq_curve(audio: np.ndarray, sample_rate: int, centers: np.ndarray, gains_db: np.ndarray, taps: int = 8191) -> np.ndarray:
    """Apply an arbitrary EQ curve with a linear-phase FIR (no phase smear)."""
    nyquist = sample_rate / 2.0
    freqs = np.concatenate(([0.0], centers[centers < nyquist], [nyquist]))
    gains = np.concatenate(([gains_db[0]], gains_db[centers < nyquist], [gains_db[-1]]))
    kernel = firwin2(taps, freqs / nyquist, db_to_gain(gains)).astype(np.float32)
    delay = (taps - 1) // 2
    n = audio.shape[-1]
    out = fftconvolve(audio, kernel[None, :], mode="full", axes=-1)[:, delay : delay + n]
    return out.astype(np.float32)


def limit(
    audio: np.ndarray,
    sample_rate: int,
    ceiling_db: float = -1.0,
    lookahead_ms: float = 5.0,
    release_ms: float = 80.0,
) -> tuple[np.ndarray, float]:
    """Look-ahead brickwall limiter on inter-sample (true) peaks.

    The gain needed at each sample is ``ceiling / true_peak``. A forward
    minimum filter over the look-ahead window followed by a moving average of
    the same length gives a smooth attack that still reaches full reduction
    by the time each peak arrives, with no signal delay. Recovery happens in
    blocks with an exponential release.

    Returns the limited audio and the deepest gain reduction in dB.
    """
    ceiling = float(db_to_gain(ceiling_db))
    peaks = true_peak_envelope(audio)
    required = np.minimum(1.0, ceiling / np.maximum(peaks, 1e-9)).astype(np.float32)
    if required.min() >= 1.0:
        return audio, 0.0

    lookahead = max(2, int(sample_rate * lookahead_ms / 1000))
    # Forward-looking: gain at n covers the peaks in [n, n + lookahead).
    ahead = minimum_filter1d(required, size=lookahead, origin=-(lookahead // 2), mode="nearest")

    block = 32
    n = ahead.size
    n_blocks = -(-n // block)
    padded = np.pad(ahead, (0, n_blocks * block - n), constant_values=1.0)
    block_min = padded.reshape(n_blocks, block).min(axis=1)
    recover = float(np.exp(-block / (sample_rate * release_ms / 1000)))
    held = np.empty_like(block_min)
    g = 1.0
    for i, target in enumerate(block_min):
        g = target if target < g else target + (g - target) * recover
        held[i] = g
    stepped = np.repeat(held, block)[:n]

    # Causal moving average: every value averaged into sample n is <= required[n].
    smooth = uniform_filter1d(stepped, size=lookahead, origin=(lookahead - 1) // 2, mode="nearest")
    out = (audio * smooth[None, :]).astype(np.float32)
    return out, float(-gain_to_db(smooth.min()))


def master(
    mix: np.ndarray,
    sample_rate: int,
    target_lufs: float = -14.0,
    ceiling_dbtp: float = -1.0,
    reference: np.ndarray | None = None,
    tonal_strength: float = 0.5,
    glue: bool = True,
    max_limiting_db: float = 6.0,
) -> tuple[np.ndarray, MasterReport]:
    """Glue, tonal balance, then hit ``target_lufs`` without exceeding ``ceiling_dbtp``.

    Gain is found iteratively because limiting lowers loudness. If reaching the
    target would need more than ``max_limiting_db`` of limiting the master
    stays quieter rather than getting crushed; the report says what it hit.
    """
    input_lufs = integrated_lufs(mix, sample_rate)
    x = glue_compress(mix, sample_rate) if glue else mix

    centers, curve = np.array([]), np.array([])
    if tonal_strength > 0:
        strength = tonal_strength if reference is None else min(1.0, tonal_strength + 0.3)
        centers, curve = tonal_correction_db(
            x, sample_rate, reference, strength=strength, max_db=4.0 if reference is None else 6.0
        )
        x = apply_eq_curve(x, sample_rate, centers, curve)

    pre_lufs = integrated_lufs(x, sample_rate)
    if not np.isfinite(pre_lufs):
        return x, MasterReport(input_lufs, pre_lufs, true_peak_db(x), 0.0, 0.0)

    # Limiting lowers loudness, so iterate. Never go past max_limiting_db:
    # stopping short of the target beats a crushed master.
    gain_db = target_lufs - pre_lufs
    best = None
    for _ in range(8):
        out, reduction = limit(x * np.float32(db_to_gain(gain_db)), sample_rate, ceiling_db=ceiling_dbtp)
        if reduction > max_limiting_db + 0.05:
            gain_db -= reduction - max_limiting_db
            continue
        best = (out, reduction, gain_db)
        error = target_lufs - integrated_lufs(out, sample_rate)
        if abs(error) < 0.1 or (error > 0 and reduction >= max_limiting_db - 0.05):
            break
        gain_db += error
    if best is None:
        best = (*limit(x * np.float32(db_to_gain(gain_db)), sample_rate, ceiling_db=ceiling_dbtp), gain_db)
    out, reduction, gain_db = best

    peak = true_peak_db(out)
    if peak > ceiling_dbtp:  # guard against residual overshoot at block edges
        out = out * np.float32(db_to_gain(ceiling_dbtp - peak - 0.01))
        peak = true_peak_db(out)

    report = MasterReport(
        input_lufs=input_lufs,
        output_lufs=integrated_lufs(out, sample_rate),
        true_peak_dbtp=peak,
        loudness_gain_db=gain_db,
        max_limiting_db=reduction,
        eq_bands_hz=[round(float(c), 1) for c in centers],
        eq_curve_db=[round(float(g), 2) for g in curve],
    )
    return out, report
