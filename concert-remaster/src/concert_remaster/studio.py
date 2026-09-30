"""Studio tracks: each song's processed instruments, kept on disk for the app's mixer.

The mixer plays the same cleaned, effect-free, studio-processed stems the
export uses, so moving a fader in the app is exactly what the exported mix
will do. Preparing them takes a little while per song, so they are cached
per song with a key over everything that shapes them; when a setting
changes, the song's tracks show as out of date until they are prepared again.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import shutil
from pathlib import Path

import numpy as np

from .analysis import integrated_lufs
from .audio_io import SAMPLE_RATE, StemWriter
from .engine import Project
from .mixing import sum_stems
from .reference import ReferenceGuide
from .render import _restore_cutoff, song_stems, track_state
from .separation import mix_stems
from .settings import Settings, to_dict

SR = SAMPLE_RATE
_SEG_KEYS = ("start", "end", "speech", "action", "fx_action")


def studio_dir(project: Project, seg_id: str) -> Path:
    return project.work_dir / "studio" / seg_id


def studio_key(project: Project, seg: dict, settings: Settings) -> str:
    """Everything that shapes a song's studio tracks (but not the mixer, which is applied live)."""
    data = to_dict(settings)
    for group in ("output", "identify", "hardware", "segmentation"):
        data.pop(group, None)
    data["mix"] = {k: v for k, v in data["mix"].items() if k not in ("stem_gains_db", "muted_stems")}
    data["master"] = {}
    effects = [e for e in project.state.get("effects") or [] if e["end"] > seg["start"] and e["start"] < seg["end"]]
    ref = seg.get("reference") or {}
    payload = {"settings": data, "seg": {k: seg.get(k) for k in _SEG_KEYS}, "effects": effects,
               "reference": ref.get("path"), "stems": mix_stems(project.state.get("aliases", {}), settings.mix.drum_kit),
               "version": 3}
    return hashlib.sha1(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:20]


def read_meta(project: Project, seg_id: str) -> dict | None:
    path = studio_dir(project, seg_id) / "meta.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def status(project: Project, seg: dict, settings: Settings | None = None) -> str:
    """'ready', 'stale' (settings changed since) or 'missing'."""
    meta = read_meta(project, seg["id"])
    if meta is None:
        return "missing"
    return "ready" if meta.get("key") == studio_key(project, seg, settings or project.settings) else "stale"


def prepare(project: Project, seg: dict, guide: ReferenceGuide | None = None) -> dict:
    """Render one song's studio tracks to disk. Returns the metadata the mixer needs."""
    settings = project.settings
    names = mix_stems(project.state["aliases"], settings.mix.drum_kit)
    start = int(seg["start"] * SR)
    stop = min(project.frames, int(seg["end"] * SR))
    song = song_stems(project, seg, names, settings, guide, _restore_cutoff(project, settings), start, stop,
                      with_crowd=True)
    # Loudness of the automatic mix as it would be heard by default, so the app can play the
    # tracks at about the exported loudness.
    audible = {n: x for n, x in song.stems.items() if not track_state(n, {}, settings)["mute"]}
    level = integrated_lufs(sum_stems(audible, song.auto_gains_db), SR) if audible else float("-inf")
    target = studio_dir(project, seg["id"])
    shutil.rmtree(target, ignore_errors=True)
    target.mkdir(parents=True)
    for name, audio in song.stems.items():
        with StemWriter(target / f"{name}.flac") as writer:
            writer.write(audio)
    meta = {
        "key": studio_key(project, seg, settings),
        "start": seg["start"],
        "end": stop / SR,
        "tracks": list(song.stems),
        "auto_gains_db": {k: round(float(v), 2) for k, v in song.auto_gains_db.items()},
        "monitor_gain_db": round(settings.master.target_lufs - level, 2) if math.isfinite(level) else 0.0,
        "empty": song.ghosts,
        "tone_matched": song.tone_matched,
        "effects_cleaned": [dataclasses.asdict(e) for e in song.events],
    }
    (target / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    return meta


def track_path(project: Project, seg_id: str, name: str) -> Path:
    return studio_dir(project, seg_id) / f"{name}.flac"


def raw_monitor_gain_db(project: Project, seg: dict, names: list[str], settings: Settings, seconds: float = 60.0) -> float:
    """Gain that brings the raw separated tracks of a song to about the export loudness."""
    from .audio_io import read_stem

    mid = (seg["start"] + seg["end"]) / 2
    a = int(max(seg["start"], mid - seconds / 2) * SR)
    b = int(min(seg["end"], mid + seconds / 2) * SR)
    parts = [read_stem(project.stem_path(n), a, b) for n in names if project.has_stem(n)]
    if not parts or b <= a:
        return 0.0
    level = integrated_lufs(np.sum(parts, axis=0), SR)
    return round(settings.master.target_lufs - level, 2) if math.isfinite(level) else 0.0
