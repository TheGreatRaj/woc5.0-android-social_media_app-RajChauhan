"""Turn separated stems and the (user-edited) timeline into finished files.

Every segment of the show becomes a *piece*:

* songs get the full studio treatment per stem, tone-matched to the studio
  original when one was found, auto-mixed and mastered;
* the artist's talk is enhanced, kept, reduced to the music under it, or cut;
* applause is kept, shortened or cut;
* stage effects (CO2 jets, fireworks, confetti) are cleaned out of every
  stem, and the song's level is kept steady;
* long silences are trimmed.

Pieces are rendered one at a time and stored on disk, then streamed into the
output files, so a 3-hour show never has to fit in memory. Each piece is
rendered with a little extra audio past its end, and neighbours that were
continuous in the show are crossfaded over that overlap, so continuous sets
flow without clicks or double beats.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import math
import re
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from .analysis import db_to_gain, integrated_lufs
from .audio_io import SAMPLE_RATE, OutputWriter, StemWriter, read_stem, resample
from .effects import EffectEvent, repair, ride_curve
from .engine import Cancelled, Project
from .mastering import limit, master
from .mixing import auto_balance, pan_stereo, sum_stems
from .reference import ReferenceGuide, balance_targets
from .restoration import restore_highs
from .separation import EXPORT_STEMS, mix_stems
from .settings import Settings
from .speech import enhance_voice, srt
from .stem_processing import process_stem, profile_for

log = logging.getLogger(__name__)

SR = SAMPLE_RATE
# Stems that carry high frequencies worth restoring on band-limited recordings.
_BRIGHT = {"vocals", "lead_vocals", "backing_vocals", "drums", "guitar", "piano", "other", "woodwinds"}
_VOICE = ("vocals", "lead_vocals", "backing_vocals")


@dataclass
class Piece:
    seg: dict
    path: Path                       # rendered audio (44.1 kHz, stored with headroom)
    start: float                     # original show time of the piece's first frame
    body_frames: int                 # frames that belong to the segment
    tail_frames: int                 # extra frames past the end, for crossfading into the next piece
    stems_dir: Path | None = None    # balanced stems for export
    report: dict = field(default_factory=dict)


def render_project(project: Project, progress: Callable[[str, float, str], None] = lambda *a: None,
                   cancel: Callable[[], bool] = lambda: False, guides: dict[str, ReferenceGuide] | None = None) -> dict:
    """Render every output the settings ask for. Returns the render report (also saved as JSON)."""
    settings = project.settings
    segments = project.state.get("segments") or [_whole_show(project)]
    names = mix_stems(project.state["aliases"], settings.mix.drum_kit)
    guides = guides or {}
    work = project.work_dir / "render"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    cutoff = _restore_cutoff(project, settings)

    pieces: list[Piece | None] = []
    fade = int(settings.crowd.fade_seconds * SR)
    total = project.frames
    for index, seg in enumerate(segments):
        if cancel():
            raise Cancelled()
        label = seg.get("title") or seg["kind"]
        progress("render", index / len(segments), f"Rendering {label}")
        nxt = segments[index + 1] if index + 1 < len(segments) else None
        tail = fade if nxt is not None and nxt.get("include", True) else 0
        pieces.append(_render_segment(project, seg, names, settings, guides.get(seg["id"]), cutoff, work, tail, total))

    video = None
    if settings.output.video and _has_video(project):
        progress("render", 0.9, "Rendering the video soundtrack")
        video = _video_pieces(project, segments, pieces, names, settings, guides, cutoff, work, total, cancel)
    progress("render", 0.95, "Writing files")
    outputs = _write_outputs(project, settings, pieces, segments)
    if video:
        progress("render", 0.97, "Writing the remastered video")
        outputs["video"] = _write_video(project, video, work)
    report = {
        "rendered": time.strftime("%Y-%m-%d %H:%M:%S"),
        "outputs": {k: [str(p) for p in v] if isinstance(v, list) else str(v) for k, v in outputs.items()},
        "songs": [dict(p.report, title=p.seg.get("title"), track=p.seg.get("track")) for p in pieces
                  if p is not None and p.seg["kind"] == "song"],
        "restored_highs_from_hz": cutoff,
    }
    (project.output_dir / "report.json").write_text(json.dumps(_clean(report), indent=2, ensure_ascii=False), encoding="utf-8")
    project.update(lambda state: state.__setitem__("render", _clean(report)))
    shutil.rmtree(work, ignore_errors=True)
    progress("render", 1.0, "Finished")
    return report


def _has_video(project: Project) -> bool:
    info = project.state.get("source_info") or {}
    if "video" not in info:  # projects made before video support
        from .audio_io import probe_source

        try:
            info = probe_source(project.source)
        except Exception:
            return False
    return bool(info.get("video")) and Path(project.source).exists()


def _video_pieces(project: Project, segments: list[dict], pieces: list, names: list[str], settings: Settings,
                  guides: dict, cutoff: float | None, work: Path, total: int, cancel) -> list[Piece]:
    """Every part of the show at its original length, for a soundtrack that stays in sync with the picture."""
    fade = int(settings.crowd.fade_seconds * SR)
    out = []
    for index, seg in enumerate(segments):
        if cancel():
            raise Cancelled()
        start, end = int(seg["start"] * SR), min(total, int(seg["end"] * SR))
        if end <= start:
            continue
        tail = min(fade, total - end) if index + 1 < len(segments) else 0
        piece = pieces[index]
        if piece is not None and piece.body_frames == end - start and piece.tail_frames == tail:
            out.append(piece)
            continue
        full = dict(seg, id=seg["id"] + "-video", include=True)
        if seg["kind"] == "talk" and (seg.get("action") or settings.speech.action) == "remove":
            full["action"] = "enhance"  # the artist is on screen: keep what they say
        piece = _render_segment(project, full, names, settings, guides.get(seg["id"]), cutoff, work, tail, total, full_length=True)
        if piece is None:  # nothing audible: keep the time with silence
            path = work / f"{full['id']}.flac"
            with StemWriter(path) as writer:
                writer.write(np.zeros((2, end - start + tail), dtype=np.float32))
            piece = Piece(full, path, start / SR, end - start, tail)
        out.append(piece)
    return out


def _write_video(project: Project, pieces: list[Piece], work: Path) -> Path:
    from .audio_io import remux_video

    soundtrack = work / "video-soundtrack.flac"
    _assemble(soundtrack, pieces, 48000, 24)
    source = Path(project.source)
    suffix = ".mp4" if source.suffix.lower() in (".mp4", ".m4v", ".mov", ".3gp") else ".mkv"
    name = re.sub(r'[<>:"/\\|?*]+', "_", project.name)
    return remux_video(source, soundtrack, project.output_dir / f"{name} - Remastered Video{suffix}")


def _whole_show(project: Project) -> dict:
    return {"id": "seg001", "kind": "song", "start": 0.0, "end": project.duration, "title": project.name,
            "track": 1, "include": True, "action": None, "speech": [], "artist": ""}


def _restore_cutoff(project: Project, settings: Settings) -> float | None:
    mode = settings.restoration.bandwidth_restore
    bandwidth = project.state.get("analysis", {}).get("bandwidth_hz") or SR / 2
    if mode == "on" or (mode == "auto" and bandwidth < 15000):
        return float(min(bandwidth, 16000))
    return None


# --- one segment ---------------------------------------------------------------


def _render_segment(project: Project, seg: dict, names: list[str], settings: Settings, guide: ReferenceGuide | None,
                    cutoff: float | None, work: Path, tail: int, total: int, full_length: bool = False) -> Piece | None:
    """Render one part of the show. ``full_length`` keeps its original duration (for video)."""
    if not seg.get("include", True):
        return None
    kind = seg["kind"]
    start = int(seg["start"] * SR)
    end = min(total, int(seg["end"] * SR))
    if end <= start:
        return None
    tail = min(tail, total - end)
    if kind in ("song", "interlude"):
        audio, stems, report = _render_song(project, seg, names, settings, guide, cutoff, start, end + tail)
    elif kind == "talk":
        audio, stems, report = _render_talk(project, seg, names, settings, start, end + tail)
    elif kind == "crowd":
        action = "keep" if full_length else seg.get("action") or settings.crowd.between_songs
        if action == "remove":
            return None
        if action == "shorten":
            end = min(end, start + int(settings.crowd.shorten_to_seconds * SR))
            tail = 0
        audio, stems, report = _render_ambience(project, names, settings, start, end + tail, settings.crowd.between_level_db)
        if action == "shorten":
            audio = _fade(audio, int(0.3 * SR), int(min(settings.crowd.fade_seconds, 2.0) * SR))
    else:  # silence: keep a short breath, trim the rest
        keep = end - start if full_length else min(end - start, int(2.0 * SR))
        end, tail = start + keep, tail if full_length else 0
        audio, stems, report = _render_ambience(project, names, settings, start, end + tail, -30.0)
    if audio is None:
        return None
    path = work / f"{seg['id']}.flac"
    with StemWriter(path) as writer:
        writer.write(audio)
    stems_dir = None
    if stems:
        stems_dir = work / seg["id"]
        for name, stem in stems.items():
            with StemWriter(stems_dir / f"{name}.flac") as writer:
                writer.write(stem)
    body = min(audio.shape[-1], end - start)
    return Piece(seg, path, start / SR, body, audio.shape[-1] - body, stems_dir, report)


def _read(project: Project, name: str, start: int, stop: int) -> np.ndarray | None:
    return read_stem(project.stem_path(name), start, stop) if project.has_stem(name) else None


_EFFECT_AMOUNT = {"remove": 1.0, "reduce": 0.5, "keep": 0.0}


def _effects(project: Project, settings: Settings, start: int, stop: int, seg: dict | None = None) -> tuple[list[EffectEvent], float]:
    """Stage effects inside ``start``..``stop`` (times relative to it) and how much to take out."""
    amount = _EFFECT_AMOUNT.get((seg or {}).get("fx_action") or settings.effects.action, 1.0)
    if amount <= 0:
        return [], 0.0
    lo, hi = start / SR, stop / SR
    events = [EffectEvent(e["kind"], max(0.0, e["start"] - lo), min(hi, e["end"]) - lo, e["strength_db"])
              for e in project.state.get("effects") or [] if e["end"] > lo and e["start"] < hi]
    return events, amount


def _clean_crowd(crowd: np.ndarray | None, events: list[EffectEvent], amount: float) -> np.ndarray | None:
    # The effects live mostly in the crowd stem; hold it at the audience's own level through them.
    return repair(crowd, SR, events, amount) if crowd is not None and events else crowd


@dataclass
class SongStems:
    """One song's instruments, cleaned and studio-processed, before any level change."""
    stems: dict[str, np.ndarray]
    auto_gains_db: dict[str, float]  # the automatic mix (the mixer's 0 dB fader position)
    events: list[EffectEvent]
    ghosts: list[str]
    tone_matched: list[str]


def song_stems(project: Project, seg: dict, names: list[str], settings: Settings, guide: ReferenceGuide | None,
               cutoff: float | None, start: int, stop: int, with_crowd: bool = False) -> SongStems:
    """Clean and process every instrument of a song and work out the automatic mix.

    The audience track is included when it will be heard (``with_crowd`` makes it
    always included, for the app's mixer).
    """
    stems = {n: _read(project, n, start, stop) for n in names}
    stems = {n: a for n, a in stems.items() if a is not None}
    events, amount = _effects(project, settings, start, stop, seg)
    with_crowd = with_crowd or not track_state("crowd", seg, settings)["mute"]
    raw_crowd = _read(project, "crowd", start, stop) if events or with_crowd else None
    processed: dict[str, np.ndarray] = {}
    tone = {}
    for name, audio in stems.items():
        if events:
            audio = repair(audio, SR, events, amount, reference=raw_crowd)
        x = process_stem(audio, SR, profile_for(name, settings.stems))
        if cutoff and name in _BRIGHT:
            x = restore_highs(x, SR, cutoff, settings.restoration.bandwidth_amount)
        ref_stem = guide.stem_for(name) if guide is not None else None
        if (ref_stem is not None and settings.reference.tone_match and settings.reference.per_stem
                and np.isfinite(integrated_lufs(ref_stem, SR))):
            from .mastering import apply_eq_curve, tonal_correction_db

            centers, curve = tonal_correction_db(x, SR, ref_stem, strength=settings.reference.tone_strength,
                                                 max_db=settings.reference.tone_max_db)
            x = apply_eq_curve(x, SR, centers, curve)
            tone[name] = [round(float(g), 2) for g in curve]
        processed[name] = x

    action = seg.get("action") or settings.speech.action
    if seg.get("speech") and action in ("remove", "music_only"):
        # Talking over the music inside a song can't be cut out, so mute just the voice there.
        envelope = np.ones(stop - start, dtype=np.float32)
        ramp = int(0.2 * SR)
        for a, b in seg["speech"]:
            lo, hi = int((a - seg["start"]) * SR), int((b - seg["start"]) * SR)
            envelope[max(0, lo):max(0, hi)] = 0.0
        envelope = np.convolve(envelope, np.ones(ramp) / ramp, mode="same").astype(np.float32)
        for name in _VOICE:
            if name in processed:
                processed[name] = processed[name] * envelope[None, : processed[name].shape[-1]]

    if guide is not None and settings.reference.balance_match and guide.stems:
        profiles = balance_targets(guide, list(processed), settings)
        strength = settings.reference.balance_strength
    else:
        profiles, strength = settings.stems, settings.mix.balance_strength
    balance = auto_balance(processed, SR, strength, settings.mix.max_adjust_db, ghost_cut_db=settings.mix.ghost_cut_db,
                           profiles=profiles)
    gains = dict(balance.gains_db)

    crowd = _clean_crowd(raw_crowd, events, amount) if with_crowd else None
    if crowd is not None and balance.reference is not None:
        crowd = process_stem(crowd, SR, profile_for("crowd", settings.stems))
        level = integrated_lufs(crowd, SR)
        if math.isfinite(level):
            ref_level = balance.loudness_lufs[balance.reference] + gains[balance.reference]
            processed["crowd"] = crowd
            gains["crowd"] = ref_level + settings.crowd.in_songs_db - level
    return SongStems(processed, gains, events, balance.ghosts, sorted(tone))


def track_state(name: str, seg: dict, settings: Settings) -> dict:
    """The user's mixer settings for one track of a song: fader offset, pan and mute."""
    gains = {**settings.mix.stem_gains_db, **(seg.get("stem_gains") or {})}
    gain = float(gains.get(name, 0.0))
    mutes = seg.get("stem_mutes") or {}
    if name in mutes:
        mute = bool(mutes[name])
    elif gain <= -60:  # older projects stored a mute as a very low gain
        mute = True
    elif name == "crowd":
        mute = not settings.crowd.keep_in_songs
    else:
        mute = name in settings.mix.muted_stems
    return {"gain_db": gain if gain > -60 else 0.0, "pan": float((seg.get("stem_pans") or {}).get(name, 0.0)), "mute": mute}


def mixdown(song: SongStems, seg: dict, settings: Settings) -> tuple[dict[str, np.ndarray], dict[str, float], float]:
    """Apply the mixer (faders, pans, mutes) and level riding. Returns placed stems, their gains, riding in dB."""
    stems: dict[str, np.ndarray] = {}
    gains: dict[str, float] = {}
    for name, audio in song.stems.items():
        state = track_state(name, seg, settings)
        if state["mute"]:
            continue
        stems[name] = pan_stereo(audio, state["pan"])
        gains[name] = song.auto_gains_db.get(name, 0.0) + state["gain_db"]
    riding = 0.0
    if stems and settings.effects.level_riding and settings.effects.riding_range_db > 0:
        curve, riding = ride_curve(sum_stems(stems, gains), SR, range_db=settings.effects.riding_range_db)
        stems = {n: (x * curve[None, : x.shape[-1]]).astype(np.float32) for n, x in stems.items()}
    return stems, gains, riding


def _render_song(project: Project, seg: dict, names: list[str], settings: Settings, guide: ReferenceGuide | None,
                 cutoff: float | None, start: int, stop: int):
    song = song_stems(project, seg, names, settings, guide, cutoff, start, stop)
    processed, gains, riding = mixdown(song, seg, settings)
    if not processed:
        return None, None, {}
    mix = sum_stems(processed, gains)
    m = settings.master
    use_ref = guide is not None and settings.reference.tone_match
    mastered, report = master(
        mix, SR, target_lufs=m.target_lufs, ceiling_dbtp=m.ceiling_dbtp,
        reference=guide.mix if use_ref else None, tonal_strength=m.tonal_strength, glue=m.glue,
        max_limiting_db=m.max_limiting_db, glue_depth_db=m.glue_depth_db,
        reference_strength=settings.reference.tone_strength if use_ref else None,
        reference_max_db=settings.reference.tone_max_db,
    )
    balanced = {n: x * np.float32(db_to_gain(gains.get(n, 0.0))) for n, x in processed.items()}
    info = {
        "loudness_lufs": report.output_lufs,
        "true_peak_dbtp": report.true_peak_dbtp,
        "limiting_db": report.max_limiting_db,
        "stem_gains_db": {k: round(v, 2) for k, v in gains.items()},
        "empty_stems": song.ghosts,
        "reference": guide.title if guide else None,
        "tone_matched_stems": song.tone_matched,
        "effects_cleaned": [e.as_dict() for e in song.events],
        "level_riding_db": round(riding, 2),
    }
    return mastered, balanced, info


def _band(project: Project, names: list[str], start: int, stop: int) -> np.ndarray | None:
    parts = [_read(project, n, start, stop) for n in names if n not in _VOICE and n != "vocals"]
    parts = [p for p in parts if p is not None]
    return np.sum(parts, axis=0) if parts else None


def _voice(project: Project, start: int, stop: int) -> np.ndarray | None:
    return _read(project, "vocals", start, stop)


def _render_talk(project: Project, seg: dict, names: list[str], settings: Settings, start: int, stop: int):
    action = seg.get("action") or settings.speech.action
    if action == "remove":
        return None, None, {}
    target = settings.master.target_lufs + settings.speech.level_lu
    voice = _voice(project, start, stop)
    band = _band(project, names, start, stop)
    crowd = _clean_crowd(_read(project, "crowd", start, stop), *_effects(project, settings, start, stop, seg))
    layers: list[np.ndarray] = []
    stems: dict[str, np.ndarray] = {}
    voice_level = None
    if voice is not None and action != "music_only":
        voice = enhance_voice(voice, SR, settings.speech) if action == "enhance" else voice
        voice_level = integrated_lufs(voice, SR)
        if math.isfinite(voice_level):
            voice = voice * np.float32(db_to_gain(target - voice_level))
            layers.append(voice)
            stems["vocals"] = voice
    if band is not None:
        level = integrated_lufs(band, SR)
        if math.isfinite(level):
            band_target = target + (settings.speech.music_under_speech_db if stems else 0.0)
            band = band * np.float32(db_to_gain(band_target - level))
            layers.append(band)
            stems["band"] = band
    if crowd is not None:
        level = integrated_lufs(crowd, SR)
        if math.isfinite(level):
            crowd = crowd * np.float32(db_to_gain(settings.master.target_lufs + settings.crowd.between_level_db - level))
            layers.append(crowd)
            stems["crowd"] = crowd
    if not layers:
        return None, None, {}
    audio, _ = limit(np.sum(layers, axis=0), SR, settings.master.ceiling_dbtp)
    return audio, stems, {"action": action}


def _render_ambience(project: Project, names: list[str], settings: Settings, start: int, stop: int, level_db: float):
    events, amount = _effects(project, settings, start, stop)
    raw_crowd = _read(project, "crowd", start, stop)
    parts = [_read(project, n, start, stop) for n in names]
    parts = [p if not events else repair(p, SR, events, amount, reference=raw_crowd) for p in parts if p is not None]
    crowd = _clean_crowd(raw_crowd, events, amount)
    parts += [crowd] if crowd is not None else []
    if not parts:
        return None, None, {}
    audio = np.sum(parts, axis=0)
    level = integrated_lufs(audio, SR)
    if math.isfinite(level):
        audio = audio * np.float32(db_to_gain(settings.master.target_lufs + level_db - level))
    audio, _ = limit(audio, SR, settings.master.ceiling_dbtp)
    return audio, None, {}


def _fade(audio: np.ndarray, fade_in: int, fade_out: int) -> np.ndarray:
    audio = audio.copy()
    n = audio.shape[-1]
    if fade_in:
        k = min(fade_in, n)
        audio[:, :k] *= np.linspace(0, 1, k, dtype=np.float32)[None]
    if fade_out:
        k = min(fade_out, n)
        audio[:, n - k:] *= np.linspace(1, 0, k, dtype=np.float32)[None]
    return audio


# --- files ---------------------------------------------------------------------------


def _file_title(seg: dict) -> str:
    title = seg.get("title") or f"Song {seg.get('track') or 0:02d}"
    artist = seg.get("artist")
    name = f"{seg.get('track') or 0:02d} - {title}" + (f" ({artist})" if artist else "")
    return re.sub(r'[<>:"/\\|?*]+', "_", name).strip(" .")[:150]


def _write_outputs(project: Project, settings: Settings, pieces: list[Piece | None], segments: list[dict]) -> dict:
    out = settings.output
    ext = out.format
    rate = out.sample_rate
    folder = project.output_dir
    folder.mkdir(parents=True, exist_ok=True)
    name = re.sub(r'[<>:"/\\|?*]+', "_", project.name)
    produced: dict[str, object] = {}

    songs = [p for p in pieces if p is not None and p.seg["kind"] == "song"]
    if out.songs:
        paths = []
        for piece in songs:
            audio = read_stem(piece.path)[:, : piece.body_frames]
            audio = _fade(audio, int(0.02 * SR), int(0.5 * SR) if piece.tail_frames else int(0.02 * SR))
            path = folder / "Songs" / f"{_file_title(piece.seg)}.{ext}"
            with OutputWriter(path, rate, out.bit_depth) as writer:
                writer.write(resample(audio, SR, rate))
            paths.append(path)
        produced["songs"] = paths

    if out.stems in ("per_song", "both"):
        dirs = []
        for piece in songs:
            if piece.stems_dir is None:
                continue
            target = folder / "Stems" / _file_title(piece.seg)
            _export_stems(piece, target, ext, rate, out.bit_depth)
            dirs.append(target)
        produced["stems"] = dirs

    timeline: list[tuple[Piece, float]] = []
    if out.full_concert:
        path = folder / f"{name} - Full Concert.{ext}"
        timeline = _assemble(path, [p for p in pieces if p is not None], rate, out.bit_depth)
        produced["full_concert"] = path
        if out.tracklist:
            produced["cue"] = _write_cue(folder / f"{name} - Full Concert.cue", path.name, timeline, project)
            produced["tracklist"] = _write_tracklist(folder / "Tracklist.txt", timeline, project)
        if out.transcript:
            entries = [dict(e, start=e["start"] - p.start + offset, end=e["end"] - p.start + offset)
                       for p, offset in timeline for e in (p.seg.get("transcript") or [])]
            if entries:
                (folder / f"{name} - Full Concert.srt").write_text(srt(entries), encoding="utf-8")
                produced["subtitles"] = folder / f"{name} - Full Concert.srt"
    if out.vibes_edition:
        vibes = [p for p in pieces if p is not None and p.seg["kind"] in ("song", "interlude", "crowd")]
        path = folder / f"{name} - Concert Vibes.{ext}"
        _assemble(path, vibes, rate, out.bit_depth)
        produced["vibes_edition"] = path
    if out.stems in ("full_concert", "both"):
        produced["full_concert_stems"] = _export_full_stems(folder / "Stems" / "Full Concert", pieces, ext, rate, out.bit_depth)
    if out.transcript:
        talks = [(s, e) for s in segments for e in (s.get("transcript") or [])]
        if talks:
            lines = [f"[{_clock(e['start'])}] {e['text']}" for _, e in talks]
            (folder / "Artist speech.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return produced


def _export_stems(piece: Piece, target: Path, ext: str, rate: int, bit_depth: int) -> None:
    files = {f.stem: read_stem(f)[:, : piece.body_frames] for f in sorted(piece.stems_dir.glob("*.flac"))}
    peak = max((float(np.abs(a).max()) for a in files.values()), default=0.0)
    mix_peak = float(np.abs(np.sum(list(files.values()), axis=0)).max()) if files else 0.0
    # One shared gain keeps the stems balanced exactly as in the mix.
    scale = min(0.89 / max(mix_peak, 1e-9), 0.89 / max(peak, 1e-9))
    for stem_name, audio in files.items():
        with OutputWriter(target / f"{stem_name}.{ext}", rate, bit_depth) as writer:
            writer.write(resample(audio * np.float32(scale), SR, rate))


def _export_full_stems(target: Path, pieces: list[Piece | None], ext: str, rate: int, bit_depth: int) -> Path:
    names = sorted({f.stem for p in pieces if p is not None and p.stems_dir for f in p.stems_dir.glob("*.flac")})
    for stem_name in names:
        with OutputWriter(target / f"{stem_name}.{ext}", rate, bit_depth) as writer:
            for piece in pieces:
                if piece is None:
                    continue
                file = piece.stems_dir / f"{stem_name}.flac" if piece.stems_dir else None
                audio = read_stem(file)[:, : piece.body_frames] if file and file.exists() else np.zeros((2, piece.body_frames), np.float32)
                writer.write(resample(audio, SR, rate))
    return target


def _assemble(path: Path, pieces: list[Piece], rate: int, bit_depth: int) -> list[tuple[Piece, float]]:
    """Stream pieces into one file, crossfading neighbours; returns each piece's start time in the file."""
    timeline = []
    position = 0
    held: np.ndarray | None = None
    with OutputWriter(path, rate, bit_depth) as writer:
        for i, piece in enumerate(pieces):
            audio = resample(read_stem(piece.path), SR, rate)
            body = int(round(piece.body_frames * rate / SR))
            nxt = pieces[i + 1] if i + 1 < len(pieces) else None
            continuous = nxt is not None and abs(nxt.start - (piece.start + piece.body_frames / SR)) < 0.05
            tail = audio.shape[-1] - body if continuous else 0
            if not continuous and piece.tail_frames:
                audio = _fade(audio[:, :body], 0, int(0.25 * rate))  # a gap follows: fade out instead
            timeline.append((piece, position / rate))
            if held is not None:
                k = min(held.shape[-1], audio.shape[-1])
                ramp = np.linspace(0, 1, k, dtype=np.float32)[None]
                audio = audio.copy()
                audio[:, :k] = held[:, :k] * (1 - ramp) + audio[:, :k] * ramp
            body_part = audio[:, : audio.shape[-1] - tail] if tail else audio
            writer.write(body_part)
            position += body_part.shape[-1]
            held = audio[:, audio.shape[-1] - tail:] if tail else None
    return timeline


def _clock(seconds: float) -> str:
    seconds = int(round(seconds))
    return f"{seconds // 3600:d}:{seconds // 60 % 60:02d}:{seconds % 60:02d}"


def _write_cue(path: Path, audio_name: str, timeline, project: Project) -> Path:
    lines = [f'TITLE "{project.name}"', f'FILE "{audio_name}" WAVE']
    for n, (piece, offset) in enumerate([t for t in timeline if t[0].seg["kind"] == "song"], 1):
        frames = int(round(offset * 75))
        lines += [f"  TRACK {n:02d} AUDIO", f'    TITLE "{piece.seg.get("title", "")}"']
        if piece.seg.get("artist"):
            lines.append(f'    PERFORMER "{piece.seg["artist"]}"')
        lines.append(f"    INDEX 01 {frames // 4500:02d}:{frames // 75 % 60:02d}:{frames % 75:02d}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _write_tracklist(path: Path, timeline, project: Project) -> Path:
    lines = [f"{project.name}", ""]
    for piece, offset in timeline:
        seg = piece.seg
        if seg["kind"] == "song":
            who = f" - {seg['artist']}" if seg.get("artist") else ""
            lines.append(f"{_clock(offset)}  {seg.get('track', 0):02d}. {seg.get('title', '')}{who}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _clean(value):
    if isinstance(value, float):
        return round(value, 2) if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if dataclasses.is_dataclass(value):
        return _clean(dataclasses.asdict(value))
    return value
