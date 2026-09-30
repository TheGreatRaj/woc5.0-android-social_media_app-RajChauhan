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
        report("segments", 1.0, f"Found {sum(s['kind'] == 'song' for s in segments)} songs")
        return segments

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
            names = mix_stems(project.state["aliases"])
            for seg in songs:
                self._check()
                if seg.get("locked"):  # the user set this one by hand
                    done += 1
                    continue
                a, b = int(seg["start"] * SAMPLE_RATE), int(seg["end"] * SAMPLE_RATE)
                mix = np.sum([read_stem(project.stem_path(n), a, b) for n in names], axis=0)
                vocals = read_stem(project.stem_path("vocals"), a, b)
                result = identify_song(mix, vocals, s.identify, library, transcriber, online,
                                       lambda m: report("identify", done / total, f"{seg['title']}: {m}"))
                seg["identification"] = {"score": result.score, "method": result.method, "message": result.message,
                                         "candidates": result.candidates[:5], "lyrics": result.lyrics[:500]}
                if result.title:
                    seg["title"], seg["artist"] = result.title, result.artist
                seg["reference"] = result.reference
                done += 1
                project.update(lambda state, seg=seg: _replace_segment(state, seg))
        report("identify", 1.0, "Songs identified")

    def export(self) -> dict:
        from .render import render_project

        project = self.project
        project.reload()
        s = project.settings
        guides = {}
        if s.reference.tone_match:
            for seg in project.state.get("segments") or []:
                ref = seg.get("reference")
                if seg["kind"] == "song" and seg.get("include", True) and ref and Path(ref.get("path", "")).exists():
                    self._check()
                    self.progress("render", 0.0, f"Preparing studio reference for {seg['title']}")
                    guides[seg["id"]] = load_guide(ref, self.backend if s.reference.per_stem else None,
                                                   s.models.instrument_model, s.reference.per_stem)
        report = render_project(project, lambda st, f, m: self.progress("render", f, m), self.cancel, guides)
        self.finish("Export finished")
        return report

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
