"""Find the songs, the artist's talking, applause and silence in a whole show.

Works on the separated stems, which makes the job much easier than on the
raw recording: the band stem says when music is playing, the vocal stem says
when someone is singing or talking, and the crowd stem says when the audience
is cheering.

* Songs are stretches where the band plays, separated by breaks. In
  continuous sets (DJ/EDM, medleys) the music never stops, so long stretches
  are split where harmony and timbre change (a novelty curve).
* Talking is told apart from singing by how the voice's pitch behaves:
  singers hold notes, speech glides constantly. Measured on real speech and
  sung vocals, the share of time spent on held notes is 0.07-0.27 for speech
  and 0.47-0.73 for singing. Rap is speech-like, so a stretch only counts as
  talk while the band is quiet.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from scipy.ndimage import binary_closing, median_filter, uniform_filter1d
from scipy.signal import find_peaks

from .audio_io import SAMPLE_RATE, read_stem, stem_frames
from .settings import SegmentationSettings

FRAME_SECONDS = 0.5
KINDS = ("song", "interlude", "talk", "crowd", "silence")


# --- features ------------------------------------------------------------------


@dataclass
class Features:
    frame_seconds: float
    band_db: np.ndarray
    vocal_db: np.ndarray
    crowd_db: np.ndarray
    hold: np.ndarray       # pitch-hold ratio of the vocal (NaN where no voice)
    chroma: np.ndarray     # (12, T) harmony of the band
    timbre: np.ndarray     # (13, T) MFCCs of the band

    @property
    def frames(self) -> int:
        return self.band_db.size

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, frame_seconds=self.frame_seconds, band_db=self.band_db, vocal_db=self.vocal_db,
                            crowd_db=self.crowd_db, hold=self.hold, chroma=self.chroma, timbre=self.timbre)

    @classmethod
    def load(cls, path: Path) -> "Features":
        data = np.load(path)
        return cls(float(data["frame_seconds"]), data["band_db"], data["vocal_db"], data["crowd_db"], data["hold"],
                   data["chroma"], data["timbre"])


def compute_features(band_path: Path | list[Path], vocal_path: Path | None, crowd_path: Path | None,
                     progress: Callable[[float], None] = lambda f: None, cancel: Callable[[], bool] = lambda: False,
                     block_seconds: float = 60.0) -> Features:
    """Frame features for the whole show, read from the stems block by block."""
    import librosa

    band_paths = band_path if isinstance(band_path, list) else [band_path]
    total = stem_frames(band_paths[0])
    hop = int(FRAME_SECONDS * SAMPLE_RATE)
    block = int(block_seconds / FRAME_SECONDS) * hop
    parts: dict[str, list[np.ndarray]] = {k: [] for k in ("band", "vocal", "crowd", "hold", "chroma", "timbre")}
    for start in range(0, total, block):
        if cancel():
            from .engine import Cancelled

            raise Cancelled()
        stop = min(total, start + block)
        band = sum(read_stem(p, start, stop) for p in band_paths)
        n_frames = max(1, (stop - start) // hop)
        parts["band"].append(_frame_db(band, hop, n_frames))
        mono = librosa.resample(band.mean(axis=0), orig_sr=SAMPLE_RATE, target_sr=22050)
        h = 11025  # 0.5 s at 22.05 kHz
        chroma = librosa.feature.chroma_stft(y=mono, sr=22050, n_fft=8192, hop_length=h, center=True)
        mfcc = librosa.feature.mfcc(y=mono, sr=22050, n_mfcc=13, n_fft=4096, hop_length=h, center=True)
        parts["chroma"].append(_fit_frames(chroma, n_frames))
        parts["timbre"].append(_fit_frames(mfcc, n_frames))
        if vocal_path is not None:
            vocal = read_stem(vocal_path, start, stop)
            parts["vocal"].append(_frame_db(vocal, hop, n_frames))
            parts["hold"].append(pitch_hold(vocal.mean(axis=0), SAMPLE_RATE, n_frames))
        if crowd_path is not None:
            parts["crowd"].append(_frame_db(read_stem(crowd_path, start, stop), hop, n_frames))
        progress(stop / total)
    frames = sum(p.size for p in parts["band"])
    empty = np.full(frames, -120.0)
    return Features(
        FRAME_SECONDS,
        np.concatenate(parts["band"]),
        np.concatenate(parts["vocal"]) if parts["vocal"] else empty,
        np.concatenate(parts["crowd"]) if parts["crowd"] else empty.copy(),
        np.concatenate(parts["hold"]) if parts["hold"] else np.full(frames, np.nan),
        np.concatenate(parts["chroma"], axis=1),
        np.concatenate(parts["timbre"], axis=1),
    )


def _frame_db(audio: np.ndarray, hop: int, n_frames: int) -> np.ndarray:
    power = np.mean(np.square(audio, dtype=np.float64), axis=0)
    power = np.pad(power, (0, max(0, n_frames * hop - power.size)))[: n_frames * hop]
    return 10.0 * np.log10(np.maximum(power.reshape(n_frames, hop).mean(axis=1), 1e-12))


def _fit_frames(features: np.ndarray, n_frames: int) -> np.ndarray:
    if features.shape[1] >= n_frames:
        return features[:, :n_frames]
    return np.pad(features, ((0, 0), (0, n_frames - features.shape[1])), mode="edge")


def pitch_hold(voice: np.ndarray, sample_rate: int, n_frames: int, window_seconds: float = 2.0) -> np.ndarray:
    """Share of voiced time spent on held notes, per analysis frame (NaN when nobody is vocalising)."""
    import librosa

    y = librosa.resample(voice, orig_sr=sample_rate, target_sr=16000)
    hop = 160  # 10 ms
    f0 = librosa.yin(y, fmin=70, fmax=900, sr=16000, frame_length=1024, hop_length=hop)
    rms = librosa.feature.rms(y=y, frame_length=640, hop_length=hop)[0]
    size = min(f0.size, rms.size)
    f0, rms = f0[:size], rms[:size]
    rms_db = 20.0 * np.log10(rms + 1e-9)
    loud = np.percentile(rms_db, 99) if size else -120.0
    active = (rms_db > loud - 35.0) & (rms_db > -60.0)
    cents = 1200.0 * np.log2(f0 / 55.0)
    steady = (np.abs(np.diff(cents, prepend=cents[:1])) < 25.0) & active
    held = np.convolve(steady.astype(float), np.ones(8), "same") >= 7.5  # 80 ms on one pitch
    window = int(window_seconds * 100)
    active_share = uniform_filter1d(active.astype(float), window, mode="nearest")
    held_share = uniform_filter1d((held & active).astype(float), window, mode="nearest")
    ratio = np.where(active_share > 0.25, held_share / np.maximum(active_share, 1e-9), np.nan)
    # Sample at the centre of each 0.5 s analysis frame.
    centres = (np.arange(n_frames) * FRAME_SECONDS + FRAME_SECONDS / 2) * 100
    return ratio[np.clip(centres.astype(int), 0, max(size - 1, 0))] if size else np.full(n_frames, np.nan)


# --- segmentation --------------------------------------------------------------


def segment(features: Features, settings: SegmentationSettings, duration: float) -> list[dict]:
    """Label the whole show as a list of non-overlapping segments."""
    fps = 1.0 / features.frame_seconds
    n = features.frames
    if n == 0:
        return []

    if not settings.split_songs:
        music_runs = [(0, n)]
    else:
        music = _music_mask(features, settings)
        music_runs = _runs(music)
        if settings.mode in ("continuous", "auto"):
            limit = None if settings.mode == "continuous" else settings.max_song_minutes * 60 * fps
            music_runs = [piece for run in music_runs for piece in _split_long(run, features, settings, limit)]

    talk = _talk_mask(features, settings) if settings.detect_speech else np.zeros(n, bool)
    crowd = _crowd_mask(features)
    labels = np.full(n, "silence", dtype=object)
    labels[crowd] = "crowd"
    labels[talk] = "talk"

    min_song = settings.min_song_seconds * fps
    song_ranges = []
    for start, stop in music_runs:
        kind = "song" if (stop - start) >= min_song or not settings.split_songs else "interlude"
        labels[start:stop] = kind
        song_ranges.append((start, stop, kind))

    segments = _absorb_short_gaps(_to_segments(labels, features.frame_seconds, duration), settings.min_break_seconds)
    # Keep song boundaries from novelty splits: _to_segments merges adjacent runs of the same kind.
    segments = _reinsert_song_splits(segments, song_ranges, features.frame_seconds)
    for seg in segments:
        if seg["kind"] in ("song", "interlude"):
            a, b = int(seg["start"] * fps), int(seg["end"] * fps)
            seg["speech"] = [[round(s / fps + seg["start"], 2), round(e / fps + seg["start"], 2)]
                             for s, e in _runs(_talk_over_music(features, settings)[a:b])
                             if (e - s) / fps >= settings.min_speech_seconds * 2]
    return number_segments(segments)


def _music_mask(features: Features, settings: SegmentationSettings) -> np.ndarray:
    reference = np.percentile(features.band_db, 95)
    music = features.band_db > reference + settings.music_threshold_db
    music = median_filter(music.astype(np.uint8), size=5).astype(bool)
    gap = max(1, int(settings.min_break_seconds / features.frame_seconds))
    # Close breaks shorter than the minimum so a quiet bar doesn't split a song.
    return binary_closing(np.pad(music, gap), structure=np.ones(gap))[gap:-gap]


def _voice_active(features: Features) -> np.ndarray:
    reference = np.percentile(features.vocal_db, 95)
    return features.vocal_db > max(reference - 30.0, -70.0)


def _speech_like(features: Features, settings: SegmentationSettings) -> np.ndarray:
    threshold = 0.25 + 0.25 * settings.speech_sensitivity  # 0.375 at the default sensitivity
    hold = np.nan_to_num(features.hold, nan=1.0)
    smoothed = uniform_filter1d(hold, 5, mode="nearest")
    return _voice_active(features) & (smoothed < threshold)


def _talk_mask(features: Features, settings: SegmentationSettings) -> np.ndarray:
    band_reference = np.percentile(features.band_db, 95)
    quiet_band = features.band_db < band_reference - 12.0
    talk = _speech_like(features, settings) & quiet_band
    return _min_runs(binary_closing(np.pad(talk, 4), structure=np.ones(4))[4:-4],
                     int(settings.min_speech_seconds / features.frame_seconds))


def _talk_over_music(features: Features, settings: SegmentationSettings) -> np.ndarray:
    band_reference = np.percentile(features.band_db, 95)
    softer_band = features.band_db < band_reference - 6.0
    return _speech_like(features, settings) & softer_band


def _crowd_mask(features: Features) -> np.ndarray:
    if np.all(features.crowd_db <= -119.0):
        return np.zeros(features.frames, bool)
    reference = np.percentile(features.crowd_db, 95)
    return median_filter((features.crowd_db > reference - 15.0).astype(np.uint8), size=3).astype(bool)


def _split_long(run: tuple[int, int], features: Features, settings: SegmentationSettings, limit: float | None) -> list[tuple[int, int]]:
    start, stop = run
    if limit is not None and stop - start <= limit:
        return [run]
    fps = 1.0 / features.frame_seconds
    min_len = int(settings.min_song_seconds * fps)
    novelty = novelty_curve(features, start, stop)
    if novelty.size < 2 * min_len:
        return [run]
    spread = np.std(novelty)
    prominence = np.median(novelty) * 0.25 + spread * (1.5 - settings.novelty_sensitivity)
    peaks, _ = find_peaks(novelty, distance=min_len, prominence=max(prominence, 1e-6))
    cuts = [p for p in peaks if min_len <= p <= novelty.size - min_len]
    edges = [start] + [start + c for c in cuts] + [stop]
    return list(zip(edges[:-1], edges[1:]))


def novelty_curve(features: Features, start: int, stop: int, context_seconds: float = 16.0) -> np.ndarray:
    """How much the music changes at each frame: distance between what came before and what comes after."""
    chroma = features.chroma[:, start:stop]
    chroma = chroma / np.maximum(np.linalg.norm(chroma, axis=0, keepdims=True), 1e-9)
    timbre = features.timbre[1:, start:stop]
    timbre = (timbre - timbre.mean(axis=1, keepdims=True)) / np.maximum(timbre.std(axis=1, keepdims=True), 1e-9)
    feat = np.vstack([chroma * 2.0, timbre * 0.3])
    k = max(2, int(context_seconds / features.frame_seconds))
    cumsum = np.concatenate([np.zeros((feat.shape[0], 1)), np.cumsum(feat, axis=1)], axis=1)
    n = feat.shape[1]
    idx = np.arange(n)
    lo, hi = np.clip(idx - k, 0, n), np.clip(idx + k, 0, n)
    before = (cumsum[:, idx] - cumsum[:, lo]) / np.maximum(idx - lo, 1)
    after = (cumsum[:, hi] - cumsum[:, idx]) / np.maximum(hi - idx, 1)
    novelty = np.linalg.norm(after - before, axis=0)
    novelty[: k // 2] = 0.0
    novelty[n - k // 2 :] = 0.0
    return uniform_filter1d(novelty, 5, mode="nearest")


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def _min_runs(mask: np.ndarray, length: int) -> np.ndarray:
    out = np.zeros_like(mask)
    for start, stop in _runs(mask):
        if stop - start >= length:
            out[start:stop] = True
    return out


def _to_segments(labels: np.ndarray, frame_seconds: float, duration: float) -> list[dict]:
    segments = []
    start = 0
    for i in range(1, labels.size + 1):
        if i == labels.size or labels[i] != labels[start]:
            segments.append({"kind": str(labels[start]), "start": round(start * frame_seconds, 2), "end": round(i * frame_seconds, 2)})
            start = i
    if segments:
        segments[-1]["end"] = round(duration, 2)
    return segments


def _absorb_short_gaps(segments: list[dict], min_seconds: float) -> list[dict]:
    """Fold brief silences (analysis-window smear at boundaries) into a neighbouring non-song segment."""
    out: list[dict] = []
    for i, seg in enumerate(segments):
        short = seg["kind"] == "silence" and seg["end"] - seg["start"] < min_seconds
        nxt = segments[i + 1] if i + 1 < len(segments) else None
        if short and out and out[-1]["kind"] not in ("song", "interlude"):
            out[-1]["end"] = seg["end"]
        elif short and nxt is not None and nxt["kind"] not in ("song", "interlude"):
            nxt["start"] = seg["start"]
        else:
            out.append(seg)
    merged: list[dict] = []
    for seg in out:
        if merged and merged[-1]["kind"] == seg["kind"] and seg["kind"] not in ("song", "interlude"):
            merged[-1]["end"] = seg["end"]
        else:
            merged.append(seg)
    return merged


def _reinsert_song_splits(segments: list[dict], song_ranges: list, frame_seconds: float) -> list[dict]:
    cuts = sorted({round(start * frame_seconds, 2) for start, _, _ in song_ranges})
    out = []
    for seg in segments:
        if seg["kind"] not in ("song", "interlude"):
            out.append(seg)
            continue
        inner = [c for c in cuts if seg["start"] < c < seg["end"]]
        edges = [seg["start"], *inner, seg["end"]]
        for a, b in zip(edges[:-1], edges[1:]):
            out.append({**seg, "start": a, "end": b})
    return out


def number_segments(segments: list[dict]) -> list[dict]:
    """Give segments stable ids and songs their track numbers; keep user-set fields."""
    track = 0
    for i, seg in enumerate(segments):
        seg["id"] = f"seg{i + 1:03d}"
        seg.setdefault("include", True)
        seg.setdefault("action", None)  # None = use the global setting
        seg.setdefault("title", "")
        seg.setdefault("artist", "")
        seg.setdefault("speech", [])
        if seg["kind"] == "song":
            track += 1
            seg["track"] = track
            if not seg["title"] or re.fullmatch(r"Song \d+", seg["title"]):
                seg["title"] = f"Song {track:02d}"
        else:
            seg["track"] = None
    return segments


def split_by_identity(start: float, end: float, windows: list[dict], min_song_seconds: float) -> list[dict] | None:
    """Split a continuous stretch where the identified track changes (DJ sets).

    Consecutive windows naming the same track form one song; a lone window
    disagreeing with both neighbours is treated as noise. The boundary is put
    halfway between the last window of one track and the first of the next,
    which lands in the DJ's crossfade. Returns None if too little was
    identified to trust (the novelty-based split is kept then).
    """
    if not windows:
        return None
    keys = [w["key"] for w in windows]
    if sum(k is not None for k in keys) < max(2, 0.4 * len(keys)):
        return None
    # Smooth: a single window that disagrees with matching neighbours takes their label.
    for i in range(1, len(keys) - 1):
        if keys[i - 1] == keys[i + 1] and keys[i] != keys[i - 1] and keys[i - 1] is not None:
            keys[i] = keys[i - 1]
    # Unknown windows join the preceding track (a track's intro/outro often doesn't match).
    for i in range(1, len(keys)):
        if keys[i] is None:
            keys[i] = keys[i - 1]
    runs: list[list[int]] = []
    for i, k in enumerate(keys):
        if runs and keys[runs[-1][0]] == k:
            runs[-1].append(i)
        else:
            runs.append([i])
    hop = windows[1]["start"] - windows[0]["start"] if len(windows) > 1 else windows[0]["end"] - windows[0]["start"]
    # Runs shorter than a song merge into the longer neighbour.
    changed = True
    while changed and len(runs) > 1:
        changed = False
        for i, run in enumerate(runs):
            if len(run) * hop < min_song_seconds:
                j = i - 1 if i == len(runs) - 1 or (i > 0 and len(runs[i - 1]) >= len(runs[i + 1])) else i + 1
                merged = sorted(runs[j] + run)
                runs[min(i, j)] = merged
                del runs[max(i, j)]
                changed = True
                break
    def centre(w):
        return (w["start"] + w["end"]) / 2

    songs = []
    for i, run in enumerate(runs):
        first, last = windows[run[0]], windows[run[-1]]
        seg_start = start if i == 0 else (centre(windows[runs[i - 1][-1]]) + centre(first)) / 2
        seg_end = end if i == len(runs) - 1 else (centre(last) + centre(windows[runs[i + 1][0]])) / 2
        best = max((windows[j] for j in run), key=lambda w: w["score"])
        songs.append({"kind": "song", "start": round(seg_start, 2), "end": round(seg_end, 2),
                      "title": best["title"], "artist": best["artist"], "reference": best["reference"],
                      "identification": {"score": best["score"], "method": "set windows",
                                         "message": f"Identified in {len(run)} windows" if best["key"] else "Not identified",
                                         "candidates": []}})
    return songs


def confident_spans(windows: list[dict], strong: float = 0.7, max_gap: int = 2) -> list[dict]:
    """Stretches of a set where one track was clearly heard.

    Windows naming the same track, with at most ``max_gap`` unknown windows between
    them, form a span; a span counts when at least two windows agree, or one matched
    very strongly (``strong``) on a track whose harmony clearly moves (a near-static
    track can match by chance). Lone weak matches are ignored as noise.
    """
    spans: list[dict] = []
    current: dict | None = None
    gap = 0
    for w in windows:
        if w["key"] is None:
            gap += 1
            if current is not None and gap > max_gap:
                spans.append(current)
                current = None
            continue
        if current is not None and current["key"] == w["key"]:
            current["windows"].append(w)
        else:
            if current is not None:
                spans.append(current)
            current = {"key": w["key"], "windows": [w]}
        gap = 0
    if current is not None:
        spans.append(current)
    out = []
    for span in spans:
        ws = span["windows"]
        best = max(ws, key=lambda w: w["score"])
        if len(ws) >= 2 or (best["score"] >= strong and best.get("motion", 1.0) >= 0.1):
            out.append({"key": span["key"], "first": (ws[0]["start"] + ws[0]["end"]) / 2,
                        "last": (ws[-1]["start"] + ws[-1]["end"]) / 2, "count": len(ws), "best": best})
    # The same track heard again right after an interruption is one span.
    merged: list[dict] = []
    for span in out:
        if merged and merged[-1]["key"] == span["key"]:
            prev = merged[-1]
            prev.update(last=span["last"], count=prev["count"] + span["count"],
                        best=max(prev["best"], span["best"], key=lambda w: w["score"]))
        else:
            merged.append(span)
    return merged


def label_by_identity(start: float, end: float, windows: list[dict], boundaries: list[float],
                      min_song_seconds: float, strong: float = 0.7) -> list[dict] | None:
    """When only part of a set was identified: name what was, keep the rest as unknown songs.

    ``boundaries`` are the existing (novelty-based) song changes. Changes are added
    between two different identified tracks that have none between them, and removed
    inside a stretch where one track was clearly playing throughout.
    """
    spans = confident_spans(windows, strong)
    if not spans:
        return None
    cuts = sorted(t for t in boundaries if start < t < end)
    for a, b in zip(spans, spans[1:]):
        if a["key"] != b["key"] and not any(a["last"] <= t <= b["first"] for t in cuts):
            cuts.append((a["last"] + b["first"]) / 2)
    # A long unidentified stretch next to a known track holds other music: end the known
    # track shortly after it was last heard (and start the next one shortly before).
    long_gap = 2 * min_song_seconds
    edges = [{"last": start - long_gap}, *spans, {"first": end + long_gap}]
    for a, b in zip(edges, edges[1:]):
        if b["first"] - a["last"] <= long_gap:
            continue
        if "key" in a and not any(a["last"] < t <= a["last"] + min_song_seconds for t in cuts):
            cuts.append(a["last"] + 10.0)
        if "key" in b and not any(b["first"] - min_song_seconds <= t < b["first"] for t in cuts):
            cuts.append(b["first"] - 10.0)
    cuts = sorted(t for t in cuts if not any(sp["first"] < t < sp["last"] for sp in spans))
    edges = [start, *cuts, end]
    parts = [[a, b] for a, b in zip(edges, edges[1:]) if b > a]
    # Parts shorter than a song join their shorter neighbour.
    while len(parts) > 1:
        short = [i for i, (a, b) in enumerate(parts) if b - a < min_song_seconds]
        if not short:
            break
        i = short[0]
        if i == 0 or (i < len(parts) - 1 and parts[i + 1][1] - parts[i + 1][0] < parts[i - 1][1] - parts[i - 1][0]):
            parts[i + 1][0] = parts[i][0]
        else:
            parts[i - 1][1] = parts[i][1]
        del parts[i]
    songs = []
    for a, b in parts:
        overlap = [(min(b, sp["last"] + 10) - max(a, sp["first"] - 10), sp) for sp in spans]
        size, span = max(overlap, key=lambda x: x[0])
        seg = {"kind": "song", "start": round(a, 2), "end": round(b, 2), "title": "", "artist": "", "reference": None,
               "identification": None}
        if size >= min(20.0, 0.3 * (b - a)):
            best = span["best"]
            seg.update(title=best["title"], artist=best["artist"], reference=best["reference"],
                       identification={"score": best["score"], "method": "set windows", "candidates": [],
                                       "message": f"Heard in {span['count']} windows"})
        songs.append(seg)
    return songs
