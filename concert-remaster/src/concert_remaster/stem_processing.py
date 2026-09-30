"""Studio-style treatment for each separated stem.

Each instrument gets the kind of chain a mix engineer would put on its track:
clean-up filtering, a downward expander to push bleed and room wash under the
music, corrective and tone EQ, compression, stereo placement and (for vocals)
de-essing plus a short plate reverb to replace the venue's reverb that was
stripped out.

Gate and compressor thresholds are relative to each stem's own loud level, so
the treatment is the same whether the separator returned the stem hot or quiet.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from pedalboard import Compressor, HighShelfFilter, LowShelfFilter, NoiseGate, PeakFilter, Pedalboard, Reverb
from scipy.signal import butter, sosfiltfilt

from .analysis import active_level_db, db_to_gain, is_mono, smoothed_envelope_db
from .restoration import highpass, lowpass, spectral_denoise


@dataclass(frozen=True)
class EQBand:
    kind: Literal["peak", "low_shelf", "high_shelf"]
    freq_hz: float
    gain_db: float
    q: float = 0.707


@dataclass(frozen=True)
class StemProfile:
    highpass_hz: float | None = 30.0
    denoise: float = 0.0
    expander_range_db: float | None = None
    expander_ratio: float = 2.0
    eq: tuple[EQBand, ...] = ()
    compress_depth_db: float | None = None
    compress_ratio: float = 3.0
    attack_ms: float = 10.0
    release_ms: float = 120.0
    deess: bool = False
    mono_below_hz: float | None = None
    width: float = 1.0
    widen_mono: bool = False
    reverb: float = 0.0
    # Target loudness relative to the lead vocal, used by the auto-mixer.
    balance_db: float = 0.0


PROFILES: dict[str, StemProfile] = {
    "vocals": StemProfile(
        highpass_hz=90.0,
        denoise=0.5,
        expander_range_db=38.0,
        eq=(
            EQBand("peak", 250.0, -2.0, 1.0),
            EQBand("peak", 3200.0, 2.0, 0.9),
            EQBand("high_shelf", 10000.0, 2.5),
        ),
        compress_depth_db=10.0,
        compress_ratio=3.0,
        attack_ms=5.0,
        release_ms=90.0,
        deess=True,
        reverb=0.08,
        balance_db=0.0,
    ),
    "drums": StemProfile(
        highpass_hz=30.0,
        expander_range_db=30.0,
        expander_ratio=1.8,
        eq=(
            EQBand("peak", 60.0, 2.0, 1.0),
            EQBand("peak", 400.0, -2.5, 1.2),
            EQBand("peak", 5000.0, 1.5, 0.8),
            EQBand("high_shelf", 12000.0, 1.5),
        ),
        compress_depth_db=8.0,
        compress_ratio=4.0,
        attack_ms=15.0,
        release_ms=120.0,
        mono_below_hz=110.0,
        width=1.1,
        balance_db=-2.0,
    ),
    "bass": StemProfile(
        highpass_hz=35.0,
        eq=(
            EQBand("peak", 80.0, 1.5, 1.0),
            EQBand("peak", 250.0, -2.0, 1.0),
            EQBand("peak", 900.0, 1.0, 1.0),
        ),
        compress_depth_db=10.0,
        compress_ratio=4.0,
        attack_ms=20.0,
        release_ms=150.0,
        mono_below_hz=150.0,
        width=0.6,
        balance_db=-4.0,
    ),
    "guitar": StemProfile(
        highpass_hz=90.0,
        eq=(
            EQBand("peak", 300.0, -1.5, 1.0),
            EQBand("peak", 3000.0, 1.5, 1.0),
            EQBand("high_shelf", 10000.0, 1.0),
        ),
        compress_depth_db=8.0,
        compress_ratio=2.5,
        attack_ms=15.0,
        release_ms=150.0,
        width=1.3,
        widen_mono=True,
        balance_db=-4.0,
    ),
    "piano": StemProfile(
        highpass_hz=50.0,
        eq=(EQBand("peak", 300.0, -1.5, 1.0), EQBand("high_shelf", 8000.0, 1.5)),
        compress_depth_db=6.0,
        compress_ratio=2.0,
        attack_ms=20.0,
        release_ms=200.0,
        width=1.2,
        widen_mono=True,
        balance_db=-5.0,
    ),
    "other": StemProfile(
        highpass_hz=50.0,
        denoise=0.25,
        eq=(EQBand("peak", 300.0, -1.5, 1.0), EQBand("high_shelf", 10000.0, 1.0)),
        compress_depth_db=6.0,
        compress_ratio=2.0,
        attack_ms=20.0,
        release_ms=200.0,
        width=1.3,
        widen_mono=True,
        balance_db=-4.0,
    ),
    "woodwinds": StemProfile(
        highpass_hz=180.0,
        eq=(EQBand("peak", 400.0, -1.5, 1.0), EQBand("peak", 2500.0, 1.0, 1.0), EQBand("high_shelf", 9000.0, 1.0)),
        compress_depth_db=6.0,
        compress_ratio=2.0,
        attack_ms=15.0,
        release_ms=150.0,
        width=1.1,
        widen_mono=True,
        reverb=0.06,
        balance_db=-6.0,
    ),
    "backing_vocals": StemProfile(
        highpass_hz=120.0,
        denoise=0.4,
        expander_range_db=35.0,
        eq=(EQBand("peak", 300.0, -2.5, 1.0), EQBand("peak", 3000.0, 1.0, 0.9), EQBand("high_shelf", 10000.0, 2.0)),
        compress_depth_db=10.0,
        compress_ratio=4.0,
        attack_ms=5.0,
        release_ms=100.0,
        deess=True,
        width=1.5,
        widen_mono=True,
        reverb=0.12,
        balance_db=-7.0,
    ),
    # Used when the band isn't split further (the "fast" presets or --stems 2).
    "instrumental": StemProfile(
        highpass_hz=30.0,
        eq=(EQBand("peak", 250.0, -1.5, 0.8), EQBand("high_shelf", 10000.0, 1.0)),
        compress_depth_db=6.0,
        compress_ratio=2.0,
        attack_ms=20.0,
        release_ms=150.0,
        mono_below_hz=100.0,
        width=1.15,
        widen_mono=True,
        balance_db=-1.0,
    ),
    # Only mixed back in when the user asks to keep some audience ambience.
    "crowd": StemProfile(highpass_hz=150.0, width=1.4, widen_mono=True, balance_db=-12.0),
}


PROFILES["lead_vocals"] = PROFILES["vocals"]
# Drum-kit pieces, for exports and for mixing the kit piece by piece: a light clean-up like
# the whole kit, and typical studio levels relative to the lead vocal.
for _piece, _balance in (("kick", -5.0), ("snare", -6.0), ("toms", -10.0), ("hihat", -12.0), ("ride", -13.0), ("crash", -12.0)):
    PROFILES[_piece] = StemProfile(highpass_hz=30.0 if _piece in ("kick", "toms") else 150.0, expander_range_db=30.0,
                                   expander_ratio=1.8, balance_db=_balance)


def profile_for(stem: str, overrides: dict[str, StemProfile] | None = None) -> StemProfile:
    if overrides and stem in overrides:
        return overrides[stem]
    return PROFILES.get(stem, PROFILES["other"])


def process_stem(audio: np.ndarray, sample_rate: int, profile: StemProfile) -> np.ndarray:
    """Run one stem through its studio chain."""
    x = np.ascontiguousarray(audio, dtype=np.float32)
    if profile.highpass_hz:
        x = highpass(x, sample_rate, profile.highpass_hz)
    if profile.denoise > 0:
        x = spectral_denoise(x, sample_rate, profile.denoise)

    loud_db = active_level_db(x, sample_rate)
    if loud_db is None:
        return x

    board = Pedalboard()
    if profile.expander_range_db is not None:
        board.append(
            NoiseGate(
                threshold_db=loud_db - profile.expander_range_db,
                ratio=profile.expander_ratio,
                attack_ms=2.0,
                release_ms=150.0,
            )
        )
    for band in profile.eq:
        board.append(_eq_plugin(band))
    if profile.compress_depth_db is not None:
        board.append(
            Compressor(
                threshold_db=loud_db - profile.compress_depth_db,
                ratio=profile.compress_ratio,
                attack_ms=profile.attack_ms,
                release_ms=profile.release_ms,
            )
        )
    if len(board):
        x = board(x, sample_rate)

    if profile.deess:
        x = deess(x, sample_rate)
    if profile.mono_below_hz:
        x = mono_below(x, sample_rate, profile.mono_below_hz)
    if profile.widen_mono and is_mono(x):
        x = pseudo_stereo(x, sample_rate)
    if profile.width != 1.0:
        x = set_width(x, profile.width)
    if profile.reverb > 0:
        x = x + plate_reverb(x, sample_rate) * profile.reverb
    return np.ascontiguousarray(x, dtype=np.float32)


def _eq_plugin(band: EQBand):
    if band.kind == "peak":
        return PeakFilter(cutoff_frequency_hz=band.freq_hz, gain_db=band.gain_db, q=band.q)
    if band.kind == "low_shelf":
        return LowShelfFilter(cutoff_frequency_hz=band.freq_hz, gain_db=band.gain_db, q=band.q)
    if band.kind == "high_shelf":
        return HighShelfFilter(cutoff_frequency_hz=band.freq_hz, gain_db=band.gain_db, q=band.q)
    raise ValueError(f"Unknown EQ band type: {band.kind}")


def _split_bands(audio: np.ndarray, sample_rate: int, split_hz: float) -> tuple[np.ndarray, np.ndarray]:
    """Zero-phase split into (low, high) that sums back to the input exactly."""
    sos = butter(4, split_hz, "lowpass", fs=sample_rate, output="sos")
    low = sosfiltfilt(sos, audio, axis=-1).astype(np.float32)
    return low, (audio - low).astype(np.float32)


def deess(
    audio: np.ndarray,
    sample_rate: int,
    split_hz: float = 5500.0,
    threshold_db: float = -12.0,
    ratio: float = 4.0,
    max_reduction_db: float = 8.0,
) -> np.ndarray:
    """Split-band de-esser: turns down only the top band during sibilants.

    Voiced singing keeps the band above ``split_hz`` 20-30 dB under the full
    signal; on "s" and "t" sounds it jumps to within a few dB. Whenever the
    top band comes closer than ``threshold_db`` to the full-band level, the
    excess is compressed at ``ratio``, up to ``max_reduction_db``.
    """
    low, high = _split_bands(audio, sample_rate, split_hz)
    high_env = smoothed_envelope_db(high, sample_rate, window_ms=5.0)
    full_env = smoothed_envelope_db(audio, sample_rate, window_ms=5.0)
    excess = high_env - full_env - threshold_db
    reduction_db = np.clip(excess * (1.0 - 1.0 / ratio), 0.0, max_reduction_db)
    gain = db_to_gain(-reduction_db).astype(np.float32)
    return (low + high * gain[None, :]).astype(np.float32)


def mono_below(audio: np.ndarray, sample_rate: int, cutoff_hz: float) -> np.ndarray:
    """Collapse everything under ``cutoff_hz`` to the centre, as releases do for kick and bass."""
    low, high = _split_bands(audio, sample_rate, cutoff_hz)
    return (high + low.mean(axis=0, keepdims=True)).astype(np.float32)


def set_width(audio: np.ndarray, width: float) -> np.ndarray:
    """Mid/side width control: 0 is mono, 1 unchanged, above 1 wider."""
    mid = 0.5 * (audio[0] + audio[1])
    side = 0.5 * (audio[0] - audio[1]) * width
    return np.stack([mid + side, mid - side]).astype(np.float32)


def pseudo_stereo(audio: np.ndarray, sample_rate: int, delay_ms: float = 14.0, amount: float = 0.35) -> np.ndarray:
    """Give a mono stem a stereo spread that still folds down to mono exactly.

    A delayed copy is added to the left and subtracted from the right, making
    complementary comb filters. Below 300 Hz nothing changes so the low end
    stays centred.
    """
    mid = audio.mean(axis=0)
    delay = int(sample_rate * delay_ms / 1000)
    delayed = np.concatenate([np.zeros(delay, dtype=np.float32), mid[: mid.size - delay]])
    diffuse = highpass(delayed[None, :], sample_rate, 300.0)[0] * amount
    return np.stack([mid + diffuse, mid - diffuse]).astype(np.float32)


def plate_reverb(audio: np.ndarray, sample_rate: int, predelay_ms: float = 30.0) -> np.ndarray:
    """Fully wet, band-limited plate-style reverb return."""
    predelay = int(sample_rate * predelay_ms / 1000)
    shifted = np.pad(audio, ((0, 0), (predelay, 0)))[:, : audio.shape[-1]]
    wet = Pedalboard([Reverb(room_size=0.55, damping=0.5, wet_level=1.0, dry_level=0.0, width=1.0)])(
        np.ascontiguousarray(shifted, dtype=np.float32), sample_rate
    )
    wet = highpass(wet, sample_rate, 300.0)
    return lowpass(wet, sample_rate, 8000.0)

