"""The whole job, step by step: what the GUI's buttons and the command line run.

    analyze = decode -> separate -> detect songs/speech -> identify & transcribe
    export  = render the edited timeline into files

Each step is resumable; re-running skips work already done with the same settings.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Callable

import numpy as np

from .audio_io import SAMPLE_RATE, read_stem
from .engine import Cancelled, Project, prepare_source, run_separation
from .identify import ReferenceLibrary, identify_song, internet_available
from .paths import models_dir
from .reference import load_guide
from .segmentation import Features, compute_features, number_segments, segment
from .separation import AudioSeparatorBackend, SeparationBackend, mix_stems
from .speech import Transcriber

log = logging.getLogger(__name__)

STAGES = ("source", "separation", "segments", "identify", "render")
STAGE_LABELS = {
    "source": "Reading the recording",
    "separation": "Separating instruments (AI)",
    "segments": "Finding songs and speech",
    "identify": "Identifying songs",
    "render": "Mixing and mastering",
    "studio": "Preparing mixer tracks",
}
# Share of an analysis run each step takes, for one overall progress bar.
_WEIGHTS = {"source": 0.02, "separation": 0.85, "segments": 0.03, "identify": 0.10}


class Job:
    """Runs steps on a project and reports progress to ``project/progress.json`` and a callback."""

    def __init__(self, project: Project, backend: SeparationBackend | None = None,
                 on_progress: Callable[[dict], None] | None = None, cancel: Callable[[], bool] = lambda: False):
        self.project = project
        self._backend = backend
        self.on_progress = on_progress
        self.cancel = cancel
        self.task: str | None = None  # what the app should resume if this run stops
        self.started = time.time()
        self._stage_started: dict[str, float] = {}
        self._fraction: dict[str, float] = {}

    @property
    def backend(self) -> SeparationBackend:
        if self._backend is None:
            s = self.project.settings
            self._backend = AudioSeparatorBackend(
                s.hardware.model_dir or models_dir(), self.project.work_dir / "tmp", device=s.hardware.device,
                half_precision=s.hardware.half_precision, roformer_overlap=s.models.roformer_overlap,
                demucs_shifts=s.models.demucs_shifts, ensemble_algorithm=s.models.vocal_ensemble_algorithm,
            )
        return self._backend

    def progress(self, stage: str, fraction: float, message: str, overall: float | None = None) -> None:
        now = time.time()
        self._stage_started.setdefault(stage, now)
        elapsed = now - self._stage_started[stage]
        eta = elapsed * (1 - fraction) / fraction if fraction > 0.01 else None
        info = {"status": "running", "task": self.task, "stage": stage, "stage_label": STAGE_LABELS.get(stage, stage),
                "fraction": round(fraction, 4), "overall": round(overall if overall is not None else fraction, 4),
                "message": message, "eta_seconds": round(eta) if eta else None, "elapsed_seconds": round(now - self.started)}
        self.project.write_progress(**info)
        if self.on_progress:
            self.on_progress(info)

    def _stage_progress(self, stage: str, base: float):
        def report(_stage: str, fraction: float, message: str) -> None:
            self.progress(stage, fraction, message, base + _WEIGHTS.get(stage, 0) * fraction)
        return report

    # --- steps -------------------------------------------------------------------

    def analyze(self, redetect: bool = False) -> None:
        base = 0.0
        prepare_source(self.project, self._stage_progress("source", base), self.cancel)
        base += _WEIGHTS["source"]
        run_separation(self.project, self.backend, self._stage_progress("separation", base), self.cancel)
        base += _WEIGHTS["separation"]
        self.detect_segments(force=redetect, base=base)
        base += _WEIGHTS["segments"]
        self.identify(base=base)
        self.finish("Analysis finished: check the songs, then export")

    def detect_segments(self, force: bool = False, base: float = 0.0) -> list[dict]:
        project = self.project
        project.reload()
        if project.state.get("segments") and not force:
            return project.state["segments"]
        report = self._stage_progress("segments", base)
        features_path = project.work_dir / "analysis" / "features.npz"
        key = str(project.state.get("stages", {}).get("instruments", {}).get("key"))
        key_file = features_path.with_suffix(".key")
        if features_path.exists() and key_file.exists() and key_file.read_text() == key:
            features = Features.load(features_path)
        else:
            aliases = project.state["aliases"]
            band = [project.stem_path(n) for n in mix_stems(aliases) if n not in ("vocals", "lead_vocals", "backing_vocals")]
            features = compute_features(band, project.stem_path("vocals"),
                                        project.stem_path("crowd") if "crowd" in aliases else None,
                                        lambda f: report("segments", 0.8 * f, "Listening for songs, talking and applause"),
                                        self.cancel)
            features.save(features_path)
            key_file.write_text(key)
        segments = segment(features, project.settings.segmentation, project.duration)
        project.update(lambda state: state.__setitem__("segments", segments))
        self.detect_effects(report)
        report("segments", 1.0, f"Found {sum(s['kind'] == 'song' for s in segments)} songs")
        return segments

    def detect_effects(self, report=None) -> list[dict]:
        """Find CO2 jets, fireworks and confetti cannons across the show (from the crowd stem)."""
        from .effects import detect_in_file

        project = self.project
        s = project.settings.effects
        kinds = {k for k, on in (("co2", s.co2), ("firework", s.fireworks), ("confetti", s.confetti)) if on}
        if not project.has_stem("crowd") or not kinds:
            events = []
        else:
            music = [project.stem_path(n) for n in mix_stems(project.state.get("aliases", {})) if project.has_stem(n)]
            events = [e.as_dict() for e in detect_in_file(
                project.stem_path("crowd"), s.sensitivity, kinds, music_paths=music,
                progress=lambda f: report and report("segments", 0.8 + 0.2 * f, "Listening for fireworks, CO2 and confetti"),
            )]
        project.update(lambda state: state.__setitem__("effects", events))
        return events

    def identify(self, base: float = 0.0, only: list[str] | None = None) -> None:
        project = self.project
        project.reload()
        s = project.settings
        report = self._stage_progress("identify", base)
        segments = project.state.get("segments") or []
        songs = [seg for seg in segments if seg["kind"] == "song" and (only is None or seg["id"] in only)]
        talks = [seg for seg in segments if seg["kind"] == "talk"] if only is None else []
        transcriber = None
        needs_whisper = (s.speech.transcribe and talks) or (s.identify.use_lyrics_search and s.identify.online)
        if needs_whisper:
            transcriber = Transcriber(s.speech.whisper_model, s.speech.language, s.hardware.device)

        total = max(1, len(songs) + len(talks))
        done = 0
        if s.speech.transcribe and talks:
            for seg in talks:
                self._check()
                report("identify", done / total, f"Transcribing what the artist says ({_clock(seg['start'])})")
                a, b = int(seg["start"] * SAMPLE_RATE), int(seg["end"] * SAMPLE_RATE)
                try:
                    seg["transcript"] = transcriber.segments(read_stem(project.stem_path("vocals"), a, b), SAMPLE_RATE, seg["start"])
                except Exception as exc:
                    log.warning("Transcription failed: %s", exc)
                    seg["transcript"] = []
                done += 1
                project.update(lambda state, seg=seg: _replace_segment(state, seg))

        if s.identify.enabled and songs:
            library = ReferenceLibrary(Path(s.identify.reference_dir) if s.identify.reference_dir else None,
                                       s.identify.library_dir or None)
            library.scan_library(lambda m: report("identify", done / total, m))
            online = s.identify.online and internet_available()
            catalog = None
            if s.identify.artist_hint.strip():
                from .identify import artist_catalog

                try:
                    catalog = artist_catalog(s.identify.artist_hint.strip(), library.root, fetch=online,
                                             progress=lambda m: report("identify", done / total, m))
                except Exception as exc:
                    log.warning("Could not fetch the artist's songs: %s", exc)
            names = mix_stems(project.state["aliases"])
            if only is None:
                songs = self._identify_continuous(project, library, catalog, names, report, done, total) or songs
                total = max(1, len(songs) + len(talks))
            for seg in songs:
                if (seg.get("identification") or {}).get("method") == "set windows" and seg.get("reference"):
                    done += 1
                    continue
                self._check()
                if seg.get("locked"):  # the user set this one by hand
                    done += 1
                    continue
                a, b = int(seg["start"] * SAMPLE_RATE), int(seg["end"] * SAMPLE_RATE)
                mix = np.sum([read_stem(project.stem_path(n), a, b) for n in names], axis=0)
                vocals = read_stem(project.stem_path("vocals"), a, b)
                result = identify_song(mix, vocals, s.identify, library, transcriber, online,
                                       lambda m: report("identify", done / total, f"{seg['title']}: {m}"), catalog)
                seg["identification"] = {"score": result.score, "method": result.method, "message": result.message,
                                         "candidates": result.candidates[:5], "lyrics": result.lyrics[:500]}
                if result.title:
                    seg["title"], seg["artist"] = result.title, result.artist
                seg["reference"] = result.reference
                done += 1
                project.update(lambda state, seg=seg: _replace_segment(state, seg))
        report("identify", 1.0, "Songs identified")

    def _identify_continuous(self, project: Project, library, catalog, names, report, done, total) -> list[dict] | None:
        """DJ sets / medleys: identify windows along each continuous stretch and split where the track changes."""
        from .identify import identify_windows
        from .segmentation import label_by_identity, split_by_identity

        s = project.settings
        segments = project.state.get("segments") or []
        libraries = [lib for lib in (catalog, library) if lib is not None and lib.entries]
        if not libraries or s.segmentation.mode == "breaks":
            return None
        regions: list[list[dict]] = []
        for seg in segments:
            if seg["kind"] == "song" and not seg.get("locked"):
                if regions and regions[-1] and abs(regions[-1][-1]["end"] - seg["start"]) < 0.6:
                    regions[-1].append(seg)
                    continue
                regions.append([seg])
            else:
                regions.append([])
        changed = False
        for region in (r for r in regions if r):
            start, end = region[0]["start"], region[-1]["end"]
            continuous = s.segmentation.mode == "continuous" or (len(region) > 1 and end - start > s.segmentation.max_song_minutes * 60)
            if not continuous:
                continue

            def read(t0, t1):
                a, b = int(t0 * SAMPLE_RATE), int(t1 * SAMPLE_RATE)
                return np.sum([read_stem(project.stem_path(n), a, b) for n in names], axis=0)

            windows = identify_windows(read, start, end, libraries, s.identify.min_match_score,
                                       progress=lambda f: report("identify", done / total, f"Following the set: {_clock(start + f * (end - start))}"))
            songs = split_by_identity(start, end, windows, s.segmentation.min_song_seconds)
            if not songs:  # too little identified to split by: name what was, keep the rest
                songs = label_by_identity(start, end, windows, [seg["start"] for seg in region[1:]],
                                          s.segmentation.min_song_seconds)
            if not songs:
                continue
            ids = {seg["id"] for seg in region}
            first = next(i for i, seg in enumerate(segments) if seg["id"] in ids)
            segments = [seg for seg in segments if seg["id"] not in ids]
            segments[first:first] = songs
            changed = True
        if not changed:
            return None
        segments = renumber(segments)
        project.update(lambda state: state.__setitem__("segments", segments))
        return [seg for seg in segments if seg["kind"] == "song"]

    def _guide(self, seg: dict, stage: str):
        """The studio original of a song, split into stems when per-stem matching is on."""
        s = self.project.settings
        ref = seg.get("reference")
        if not (s.reference.tone_match and ref and Path(ref.get("path", "")).exists()):
            return None
        self._check()
        self.progress(stage, self._fraction.get(stage, 0.0), f"Preparing studio reference for {seg.get('title') or seg['id']}")
        return load_guide(ref, self.backend if s.reference.per_stem else None, s.models.instrument_model, s.reference.per_stem)

    def export(self) -> dict:
        from .render import render_project

        project = self.project
        project.reload()
        guides = {}
        for seg in project.state.get("segments") or []:
            if seg["kind"] == "song" and seg.get("include", True):
                guide = self._guide(seg, "render")
                if guide is not None:
                    guides[seg["id"]] = guide
        report = render_project(project, lambda st, f, m: self.progress("render", f, m), self.cancel, guides)
        self.finish("Export finished")
        return report

    def prepare_studio(self, only: list[str] | None = None) -> list[str]:
        """Render the mixer tracks of every song (or ``only`` these) that needs it."""
        from . import studio

        project = self.project
        project.reload()
        songs = [seg for seg in project.state.get("segments") or []
                 if seg["kind"] in ("song", "interlude") and (only is None or seg["id"] in only)
                 and (only is not None or seg.get("include", True))]
        todo = [seg for seg in songs if studio.status(project, seg) != "ready"]
        done: list[str] = []
        for i, seg in enumerate(todo):
            self._check()
            self._fraction["studio"] = i / max(1, len(todo))
            guide = self._guide(seg, "studio")
            self.progress("studio", i / max(1, len(todo)), f"Mixer tracks for {seg.get('title') or seg['id']} ({i + 1}/{len(todo)})")
            studio.prepare(project, seg, guide)
            done.append(seg["id"])
        return done

    def run_all(self) -> dict:
        self.analyze()
        return self.export()

    def finish(self, message: str) -> None:
        self.project.write_progress(status="done", stage="done", fraction=1.0, overall=1.0, message=message,
                                    elapsed_seconds=round(time.time() - self.started))

    def _check(self) -> None:
        if self.cancel():
            raise Cancelled()


def _replace_segment(state: dict, seg: dict) -> None:
    state["segments"] = [seg if s.get("id") == seg["id"] else s for s in state.get("segments", [])]


def _clock(seconds: float) -> str:
    seconds = int(seconds)
    return f"{seconds // 3600}:{seconds // 60 % 60:02d}:{seconds % 60:02d}"


def renumber(segments: list[dict]) -> list[dict]:
    """After the user edits the timeline: re-number songs, keep their titles."""
    for seg in segments:
        seg.pop("track", None)
    return number_segments(segments)
