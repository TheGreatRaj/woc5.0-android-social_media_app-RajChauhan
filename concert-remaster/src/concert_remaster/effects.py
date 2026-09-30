"""Stage special effects: CO2 / smoke jets, fireworks and confetti cannons.

After AI separation, sounds of the venue (audience, pyrotechnics, CO2 cannons)
mostly land in the crowd stem, with some residue in the music stems. Effects
are found there by their shape:

* CO2 / smoke jets: a sudden, very noise-like hiss with a *smooth* envelope
  (applause is just as noise-like but made of dense claps, so its envelope is
  rough).
* Fireworks: a sudden low-frequency boom with a long decay, often followed by
  crackle.
* Confetti cannons: an extremely sharp broadband pop, then paper rustle.

Removal repairs each stem spectrally: the part of a music stem that follows
the crowd stem during an event is cancelled, then every frequency is limited
to the level the stem had just before and after it, so the added noise
energy goes but the music (and its loudness) stays. Nothing is muted, which
keeps the level steady through the effect; :func:`ride_levels` then evens out
what level changes remain.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.ndimage import maximum_filter1d, median_filter, percentile_filter, uniform_filter, uniform_filter1d

from .restoration import istft, stft

HOP_S = 0.02


@dataclass
class EffectEvent:
    kind: str       # "co2", "firework", "confetti"
    start: float    # seconds
    end: float
    strength_db: float

    def as_dict(self) -> dict:
        return {k: round(v, 2) if isinstance(v, float) else v for k, v in asdict(self).items()}


def _frames(audio: np.ndarray, sample_rate: int, n_fft: int = 2048):
    hop = int(sample_rate * HOP_S)
    mono = np.asarray(audio, dtype=np.float32).mean(axis=0)
    pad = np.pad(mono, (n_fft // 2, n_fft // 2))
    count = 1 + (pad.size - n_fft) // hop
    frames = np.lib.stride_tricks.sliding_window_view(pad, n_fft)[::hop][:count]
    spec = np.abs(np.fft.rfft(frames * np.hanning(n_fft).astype(np.float32), axis=-1)) ** 2
    freqs = np.fft.rfftfreq(n_fft, 1 / sample_rate)
    return spec, freqs, mono, hop


def detect_effects(ambience: np.ndarray, sample_rate: int, sensitivity: float = 0.5,
                   music: np.ndarray | None = None) -> list[EffectEvent]:
    """Find effect bursts in the crowd/ambience stem.

    ``music`` (the other stems summed) guards against music that bled into the crowd
    stem: kick drums look like booms and noise risers like jets there, but they rise
    and fall together with the music, while a real effect jumps out of it.
    """
    spec, freqs, mono, hop = _frames(ambience, sample_rate)
    if spec.shape[0] < 50:
        return []
    band = (freqs >= 300) & (freqs <= 12000)
    low = freqs < 150
    power = spec[:, band].mean(axis=1) + 1e-12
    energy_db = 10 * np.log10(spec.sum(axis=1) + 1e-12)
    low_db = 10 * np.log10(spec[:, low].sum(axis=1) + 1e-12)
    flatness = np.exp(np.mean(np.log(spec[:, band] + 1e-12), axis=1)) / power
    # Baselines: the stem's usual floor over the surrounding 8 s. A low percentile rather
    # than the median, so a long effect doesn't raise its own baseline.
    win = int(8.0 / HOP_S)
    base = percentile_filter(energy_db, 20, size=win, mode="nearest")
    base_low = percentile_filter(low_db, 20, size=win, mode="nearest")
    # Envelope roughness in 5 ms steps: applause flickers, a gas jet doesn't.
    fine = int(sample_rate * 0.005)
    env = np.sqrt(uniform_filter1d(mono**2, fine)[::fine] + 1e-12)
    env_db = 20 * np.log10(env)
    step = hop // fine
    rough_fine = np.abs(np.diff(env_db, prepend=env_db[:1]))
    rough = uniform_filter1d(rough_fine, 20)[::step][: energy_db.size]
    rough = np.pad(rough, (0, max(0, energy_db.size - rough.size)), mode="edge")

    surge = 6.0 + 8.0 * (1.0 - sensitivity)  # dB above the local baseline
    events: list[EffectEvent] = []

    # CO2 / smoke: loud, flat, smooth, lasting at least 0.4 s.
    jet = (energy_db - base > surge) & (flatness > 0.3) & (rough < 1.2)
    events += _events(jet, "co2", energy_db - base, min_s=0.4, pad_before=0.1, pad_after=0.3)

    # Fireworks: a sudden low boom well above the low-end baseline, decaying over >0.3 s. The
    # boom must carry a good share of all the sound: a cheer swelling up raises the lows too,
    # but its energy stays in the voice range. (The low band has few bins, so it is smoothed
    # a little first to keep random flicker in a quiet low end from looking like a rise.)
    low_db = uniform_filter1d(low_db, 3, mode="nearest")
    low_share = low_db - energy_db
    rise = low_db - np.concatenate([np.full(3, low_db[0]), low_db[:-3]])
    boom_onsets = np.flatnonzero((rise > 12) & (low_db - base_low > surge + 4))
    last = -10**9
    for i in boom_onsets:  # validate first, then thin, so a rejected wobble can't hide the real onset
        if i - last <= int(0.5 / HOP_S) or np.max(low_share[i : i + 10]) < -10:
            continue
        peak = i + int(np.argmax(low_db[i : i + 10]))
        tail = low_db[peak : peak + int(2.0 / HOP_S)]
        decay = int(np.argmax(tail < low_db[peak] - 15)) if np.any(tail < low_db[peak] - 15) else tail.size
        if decay * HOP_S >= 0.3:
            events.append(EffectEvent("firework", max(0.0, (i - 3) * HOP_S), (peak + decay) * HOP_S + 0.5,
                                      float(low_db[peak] - base_low[peak])))
            last = i

    # Confetti: a near-instant broadband pop that dies away at once (a crowd suddenly
    # getting loud jumps just as fast, but stays loud).
    jump = energy_db - np.concatenate([[energy_db[0]], energy_db[:-1]])
    pops = np.flatnonzero((jump > surge + 6) & (flatness > 0.25) & (energy_db - base > surge + 4))
    last = -10**9
    for i in pops:
        after = energy_db[i + 5 : i + 15]
        if i - last <= int(0.5 / HOP_S) or (after.size and np.median(after) > energy_db[i] - 6):
            continue
        events.append(EffectEvent("confetti", max(0.0, (i - 2) * HOP_S), (i + int(1.2 / HOP_S)) * HOP_S, float(energy_db[i] - base[i])))
        last = i

    if music is not None and events:
        m_spec, _, _, _ = _frames(music, sample_rate)
        n = min(m_spec.shape[0], spec.shape[0])
        m_energy = 10 * np.log10(m_spec[:n].sum(axis=1) + 1e-12)
        m_low = uniform_filter1d(10 * np.log10(m_spec[:n, low].sum(axis=1) + 1e-12), 3, mode="nearest")
        rel = energy_db[:n] - m_energy
        rel_low = low_db[:n] - m_low
        rel_base = percentile_filter(rel, 50, size=win, mode="nearest")
        rel_low_base = percentile_filter(rel_low, 50, size=win, mode="nearest")
        need = 6.0 + 4.0 * (1.0 - sensitivity)
        kept = []
        for e in events:
            a, b = int(e.start / HOP_S), min(n, int(e.end / HOP_S) + 1)
            if b <= a:
                continue
            r, rb, level = (rel_low, rel_low_base, low_db) if e.kind == "firework" else (rel, rel_base, energy_db)
            jump = r[a:b] - rb[a:b]
            # A boom or pop must stand out from the music at its loudest moment; a jet all the way through.
            peak = int(np.argmax(level[a:min(b, a + 15)]))
            stands = np.median(jump) if e.kind == "co2" else jump[peak]
            if float(stands) > need:
                kept.append(e)
        events = kept

    return _merge(sorted(events, key=lambda e: e.start))


def _events(mask: np.ndarray, kind: str, excess: np.ndarray, min_s: float, pad_before: float, pad_after: float) -> list[EffectEvent]:
    mask = median_filter(mask.astype(np.uint8), size=5).astype(bool)
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    out = []
    for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        if (b - a) * HOP_S >= min_s:
            out.append(EffectEvent(kind, max(0.0, a * HOP_S - pad_before), b * HOP_S + pad_after, float(np.max(excess[a:b]))))
    return out


def _merge(events: list[EffectEvent]) -> list[EffectEvent]:
    merged: list[EffectEvent] = []
    for e in events:
        if merged and e.start <= merged[-1].end:
            last = merged[-1]
            merged[-1] = EffectEvent(last.kind if last.strength_db >= e.strength_db else e.kind, last.start,
                                     max(last.end, e.end), max(last.strength_db, e.strength_db))
        else:
            merged.append(e)
    return merged


def repair(audio: np.ndarray, sample_rate: int, events: list[EffectEvent], amount: float = 1.0,
           reference: np.ndarray | None = None, context_s: float = 2.5, margin_db: float = 3.0,
           n_fft: int = 2048, block_frames: int = 48) -> np.ndarray:
    """Remove effect sound from a music stem during ``events``, in two steps.

    1. Cancellation: separation splits an effect between the crowd stem
       (``reference``) and the music stems, so the leftover in a music stem is a
       filtered copy of what the crowd stem holds at the same moment. Per
       frequency, that coherent part is estimated over the event and
       subtracted, weighted by how strongly the two agree.
    2. Clamping: every frequency is then limited to the music's own level just
       before and after the event (a high percentile) plus ``margin_db``.

    Gains are smoothed and ramped, so there is no level dip and nothing changes
    outside events. ``amount`` 1 = remove, 0.5 = halfway, 0 = leave alone.
    """
    if amount <= 0 or not events:
        return audio
    hop = n_fft // 4
    spec = stft(audio, n_fft, hop)
    fps = sample_rate / hop
    if reference is not None:
        ref = stft(reference, n_fft, hop)
        original = np.abs(spec)
        ctx = int(context_s * fps)
        for e in events:
            a, b = int(e.start * fps), min(spec.shape[2], int(e.end * fps) + 1)
            around = np.r_[max(0, a - ctx):a, b:min(spec.shape[2], b + ctx)]
            if b - a < 4 or around.size < 4:
                continue
            s_part, r_part = spec[:, :, a:b], ref[:, :, a:b]
            # How the stem follows the crowd stem, estimated in short overlapping blocks
            # (masks drift during an event).
            k = min(block_frames, b - a)
            cross = uniform_filter1d(s_part * np.conj(r_part), k, axis=2, mode="nearest")
            r_power = uniform_filter1d(np.abs(r_part) ** 2, k, axis=2, mode="nearest") + 1e-12
            s_power = uniform_filter1d(np.abs(s_part) ** 2, k, axis=2, mode="nearest") + 1e-12
            coherence = np.abs(cross) ** 2 / (r_power * s_power)
            weight = np.clip((coherence - 0.1) / 0.4, 0.0, 1.0) * amount
            # A stem never holds more of an effect than the crowd stem itself; this also stops
            # music that bleeds into the crowd stem from being read as a (huge) leak.
            coef = cross / r_power
            coef = coef * np.minimum(1.0, 1.0 / np.maximum(np.abs(coef), 1e-12))
            # The crowd stem also carries some of the music. Only its surge above its usual
            # level is the effect, so only that part is cancelled. (A high percentile: music in
            # the crowd stem comes and goes with the notes and must not count as a surge.)
            r_usual = np.percentile(np.abs(ref[:, :, around]), 90, axis=2, keepdims=True)
            surge = r_part * np.clip(1.0 - 1.4 * r_usual / np.maximum(np.abs(r_part), 1e-12), 0.0, 1.0)
            ramp = np.minimum(1.0, np.minimum(np.arange(b - a) + 1, np.arange(b - a)[::-1] + 1) / 4.0)
            fixed = s_part - coef * surge * weight * ramp[None, None, :]
            # Guard: cancelling may only take energy away, and never below the stem's usual level.
            before = original[:, :, a:b]
            floor = np.median(original[:, :, around], axis=2, keepdims=True) * 10 ** (-3 / 20)
            size = np.abs(fixed)
            target = np.clip(size, np.minimum(before, floor), before)
            phase = np.where(size > 1e-9, fixed / np.maximum(size, 1e-12), s_part / np.maximum(before, 1e-12))
            spec[:, :, a:b] = phase * target
    mag = np.abs(spec).mean(axis=0)  # linked stereo
    frames = mag.shape[1]
    gain = np.ones_like(mag)
    for e in events:
        a, b = int(e.start * fps), min(frames, int(e.end * fps) + 1)
        if b <= a:
            continue
        ctx = int(context_s * fps)
        context = np.concatenate([mag[:, max(0, a - ctx):a], mag[:, b:min(frames, b + ctx)]], axis=1)
        if context.shape[1] < 4:
            continue
        # Lenient to notes moving by a bin or two: the ceiling is the loudest nearby bin's level.
        ceiling = maximum_filter1d(np.percentile(context, 90, axis=1), 5, mode="nearest")[:, None] * 10 ** (margin_db / 20)
        local = np.minimum(1.0, ceiling / np.maximum(mag[:, a:b], 1e-12))
        gain[:, a:b] = np.minimum(gain[:, a:b], local)
    gain = uniform_filter(gain, size=(3, 5), mode="nearest")
    gain = np.minimum(1.0, gain ** amount)
    out = istft(spec * gain[None].astype(np.float32), n_fft, hop, audio.shape[-1])
    return out


def ride_levels(audio: np.ndarray, sample_rate: int, range_db: float = 4.0, window_s: float = 3.0,
                trend_s: float = 30.0, deadband_db: float = 2.0, strength: float = 0.7) -> tuple[np.ndarray, float]:
    """Apply :func:`ride_curve`; returns the audio and the largest correction in dB."""
    curve, largest = ride_curve(audio, sample_rate, range_db, window_s, trend_s, deadband_db, strength)
    return (audio * curve[None, :]).astype(np.float32), largest


def ride_curve(audio: np.ndarray, sample_rate: int, range_db: float = 4.0, window_s: float = 3.0,
               trend_s: float = 30.0, deadband_db: float = 2.0, strength: float = 0.7) -> tuple[np.ndarray, float]:
    """Even out level dips and bumps (a phone's auto-gain, removed effects) without flattening the music.

    Short-term loudness is compared with the slow trend of the song; only
    deviations beyond ``deadband_db`` are corrected, partly (``strength``), by
    at most ``range_db``, with slow smoothing, so builds and breakdowns survive.
    Returns a per-sample gain curve and the largest correction in dB.
    """
    hop = int(0.5 * sample_rate)
    mono = np.mean(np.square(audio, dtype=np.float64), axis=0)
    n = mono.size // hop
    if n < 8:
        return np.ones(audio.shape[-1], dtype=np.float32), 0.0
    blocks = mono[: n * hop].reshape(n, hop).mean(axis=1)
    short = 10 * np.log10(uniform_filter1d(blocks, max(1, int(window_s / 0.5)), mode="nearest") + 1e-12)
    active = short > np.percentile(short, 95) - 30
    trend = median_filter(np.where(active, short, np.median(short[active]) if active.any() else short),
                          size=max(3, int(trend_s / 0.5)), mode="nearest")
    deviation = short - trend
    excess = np.sign(deviation) * np.maximum(np.abs(deviation) - deadband_db, 0.0)
    correction = np.clip(-excess * strength, -range_db, range_db)
    correction = np.where(active, correction, 0.0)
    correction = uniform_filter1d(correction, max(1, int(4.0 / 0.5)), mode="nearest")
    curve = np.interp(np.arange(audio.shape[-1]), (np.arange(n) + 0.5) * hop, 10 ** (correction / 20))
    return curve.astype(np.float32), float(np.max(np.abs(correction)))


def detect_in_file(path, sensitivity: float = 0.5, kinds: set[str] | None = None, block_s: float = 60.0,
                   overlap_s: float = 10.0, progress=lambda fraction: None, music_paths=()) -> list[EffectEvent]:
    """Run :func:`detect_effects` over a whole stored crowd stem in overlapping blocks.

    ``music_paths``: the music stems, summed as the guard against bleed.
    """
    from .audio_io import SAMPLE_RATE, read_stem, stem_frames

    total = stem_frames(path)
    events: list[EffectEvent] = []
    step = int(block_s * SAMPLE_RATE)
    for start in range(0, total, step):
        lo = max(0, start - int(overlap_s * SAMPLE_RATE))
        hi = min(total, start + step + int(overlap_s * SAMPLE_RATE))
        offset = lo / SAMPLE_RATE
        music = np.sum([read_stem(m, lo, hi) for m in music_paths], axis=0) if music_paths else None
        for e in detect_effects(read_stem(path, lo, hi), SAMPLE_RATE, sensitivity, music):
            e = EffectEvent(e.kind, e.start + offset, e.end + offset, e.strength_db)
            # Keep events that start inside this block's own span; the overlap only gives context.
            if start / SAMPLE_RATE <= e.start < (start + step) / SAMPLE_RATE and (kinds is None or e.kind in kinds):
                events.append(e)
        progress(min(1.0, (start + step) / total))
    return _merge(sorted(events, key=lambda e: e.start))
