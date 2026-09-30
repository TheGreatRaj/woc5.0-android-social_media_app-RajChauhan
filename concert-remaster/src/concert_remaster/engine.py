"""Long-recording engine: one project folder per concert, resumable at every step.

A 3-hour show is ~3.8 GB per stem as float audio, so nothing here holds a
whole recording in memory. The source is decoded in blocks, each AI pass runs
over overlapping chunks that are crossfaded back together, and every stem
lives on disk as FLAC. Each chunk is cached as soon as it finishes: a crash,
a closed laptop or a cancelled job picks up where it stopped.

Project layout::

    <project>/
        project.json          settings, stage status, songs, segments
        progress.json         live progress of a running job
        work/stems/*.flac     full-length stems
        work/chunks/<pass>/   per-chunk results while a pass runs
        output/               finished files
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Callable

import numpy as np

from . import settings as settings_mod
from .audio_io import SAMPLE_RATE, StemWriter, iter_stem, probe_source, read_stem, stem_frames, stream_decode
from .paths import projects_dir
from .restoration import RumbleFilter, clip_levels, declip, estimate_bandwidth, find_ceilings, plateau_counts
from .separation import PassSpec, SeparationBackend, build_passes, stem_aliases
from .settings import Settings

log = logging.getLogger(__name__)

ProgressFn = Callable[[str, float, str], None]


class Cancelled(Exception):
    """Raised when the user stops a running job."""


def _noop_progress(stage: str, fraction: float, message: str) -> None:
    pass


# --- project -----------------------------------------------------------------


class Project:
    FILE = "project.json"

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.state = json.loads((self.root / self.FILE).read_text(encoding="utf-8"))

    @classmethod
    def create(cls, source: str | Path, projects_root: str | Path | None = None, settings: Settings | None = None,
               name: str | None = None) -> "Project":
        source = Path(source).resolve()
        if not source.is_file():
            raise FileNotFoundError(f"No such file: {source}")
        base = Path(projects_root) if projects_root else projects_dir()
        name = name or source.stem
        root = base / _safe_name(name)
        suffix = 2
        while (root / cls.FILE).exists():
            root = base / f"{_safe_name(name)} ({suffix})"
            suffix += 1
        root.mkdir(parents=True, exist_ok=True)
        state = {
            "version": 2,
            "name": name,
            "source": str(source),
            "created": _dt.datetime.now().isoformat(timespec="seconds"),
            "source_info": probe_source(source),
            "settings": settings_mod.to_dict(settings or Settings()),
            "stages": {},
            "aliases": {},
            "analysis": {},
            "segments": [],
        }
        (root / cls.FILE).write_text(json.dumps(state, indent=2), encoding="utf-8")
        return cls(root)

    # state ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return self.state["name"]

    @property
    def source(self) -> Path:
        return Path(self.state["source"])

    @property
    def settings(self) -> Settings:
        return settings_mod.from_dict(self.state.get("settings"))

    def set_settings(self, settings: Settings) -> None:
        self.update(lambda state: state.__setitem__("settings", settings_mod.to_dict(settings)))

    def reload(self) -> None:
        self.state = json.loads((self.root / self.FILE).read_text(encoding="utf-8"))

    def save(self) -> None:
        tmp = self.root / (self.FILE + ".tmp")
        tmp.write_text(json.dumps(self.state, indent=2), encoding="utf-8")
        os.replace(tmp, self.root / self.FILE)

    def update(self, mutate: Callable[[dict], None]) -> None:
        """Re-read, change and save, so the GUI and a worker process don't overwrite each other."""
        lock = self.root / ".lock"
        deadline = time.monotonic() + 10
        while True:
            try:
                fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                break
            except FileExistsError:
                if time.monotonic() > deadline:
                    lock.unlink(missing_ok=True)  # stale lock from a killed process
                    deadline = time.monotonic() + 10
                    continue
                time.sleep(0.05)
        try:
            self.reload()
            mutate(self.state)
            self.save()
        finally:
            os.close(fd)
            lock.unlink(missing_ok=True)

    # paths ------------------------------------------------------------------

    @property
    def work_dir(self) -> Path:
        return self.root / "work"

    @property
    def stems_dir(self) -> Path:
        return self.work_dir / "stems"

    @property
    def output_dir(self) -> Path:
        return self.root / "output"

    def stem_path(self, name: str) -> Path:
        file = self.state.get("aliases", {}).get(name, name)
        return self.stems_dir / f"{file}.flac"

    def has_stem(self, name: str) -> bool:
        return self.stem_path(name).is_file()

    @property
    def frames(self) -> int:
        return stem_frames(self.stem_path("source"))

    @property
    def duration(self) -> float:
        return self.frames / SAMPLE_RATE

    def stage_done(self, name: str, key: str) -> bool:
        stage = self.state.get("stages", {}).get(name)
        return bool(stage and stage.get("done") and stage.get("key") == key)

    def mark_stage(self, name: str, key: str, **info) -> None:
        def mutate(state):
            state.setdefault("stages", {})[name] = {"key": key, "done": True, "finished": _dt.datetime.now().isoformat(timespec="seconds"), **info}
        self.update(mutate)

    def write_progress(self, **info) -> None:
        tmp = self.root / "progress.json.tmp"
        tmp.write_text(json.dumps({"time": time.time(), **info}), encoding="utf-8")
        os.replace(tmp, self.root / "progress.json")

    def read_progress(self) -> dict:
        try:
            return json.loads((self.root / "progress.json").read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            return {}


def _safe_name(name: str) -> str:
    cleaned = "".join(c if c.isalnum() or c in " -_.()&'" else "_" for c in name).strip(" .")
    return cleaned[:120] or "concert"


def _key(*parts) -> str:
    return hashlib.sha1(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()[:16]


# --- stage 1: decode, declip, rumble ------------------------------------------


def source_key(project: Project) -> str:
    source = project.source
    stat = source.stat()
    r = project.settings.restoration
    return _key("source", str(source), stat.st_size, int(stat.st_mtime), r.declip, r.declip_threshold, r.rumble_hz)


def prepare_source(project: Project, progress: ProgressFn = _noop_progress, cancel: Callable[[], bool] = lambda: False) -> None:
    """Decode the recording to ``stems/source.flac``: declipped, rumble-free, 44.1 kHz stereo."""
    key = source_key(project)
    if project.stage_done("source", key) and project.has_stem("source"):
        return
    r = project.settings.restoration
    duration = project.state.get("source_info", {}).get("duration") or 0.0
    total_guess = max(1, int(duration * SAMPLE_RATE))
    raw_path = project.stems_dir / "source_raw.flac"
    block = 60 * SAMPLE_RATE

    # Pass A: decode to disk and find the ceilings.
    ceilings = None
    samples = []
    with StemWriter(raw_path) as writer:
        for audio in stream_decode(project.source, SAMPLE_RATE, block_seconds=60):
            _check(cancel)
            block_ceilings = find_ceilings(audio)
            ceilings = block_ceilings if ceilings is None else np.maximum(ceilings, block_ceilings)
            writer.write(audio)
            if len(samples) < 40 and (writer.frames // block) % 3 == 0:
                samples.append(audio[:, : 10 * SAMPLE_RATE])
            progress("source", 0.4 * min(1.0, writer.frames / total_guess), "Decoding the recording")
    total = stem_frames(raw_path)
    if total == 0:
        raise ValueError(f"{project.source.name} contains no audio")

    # Pass B: is it clipped? Count flat-topped plateaus against the global ceilings.
    levels = np.full(ceilings.shape, np.nan)
    if r.declip:
        counts = np.zeros(ceilings.shape, dtype=np.int64)
        for offset, audio in iter_stem(raw_path, block):
            _check(cancel)
            counts += plateau_counts(audio, ceilings)
            progress("source", 0.4 + 0.2 * offset / total, "Checking for clipping")
        levels = clip_levels(ceilings, counts)

    # Pass C: repair clipped peaks (with margins so splines see across block edges), filter rumble.
    margin = 4096
    repaired = 0
    rumble = RumbleFilter(SAMPLE_RATE, r.rumble_hz)
    with StemWriter(project.stems_dir / "source.flac") as writer:
        for start in range(0, total, block):
            _check(cancel)
            stop = min(total, start + block)
            lo, hi = max(0, start - margin), min(total, stop + margin)
            audio = read_stem(raw_path, lo, hi)
            if np.isfinite(levels).any():
                audio, n = declip(audio, threshold=r.declip_threshold, levels=levels)
                repaired += n
            writer.write(rumble.process(audio[:, start - lo : start - lo + (stop - start)]))
            progress("source", 0.6 + 0.4 * stop / total, "Repairing clipped peaks" if np.isfinite(levels).any() else "Filtering rumble")
    raw_path.unlink(missing_ok=True)

    sample = np.concatenate(samples, axis=1) if samples else read_stem(project.stem_path("source"), 0, 60 * SAMPLE_RATE)
    side = sample[0] - sample[1]
    analysis = {
        "duration": total / SAMPLE_RATE,
        "declipped_samples": int(repaired),
        "clipped_channels": int(np.isfinite(levels).any(axis=1).sum()),
        "bandwidth_hz": estimate_bandwidth(sample, SAMPLE_RATE),
        "mono": bool(np.mean(side**2) < 1e-4 * max(np.mean(sample**2), 1e-12)),
    }
    project.update(lambda state: state.setdefault("analysis", {}).update(analysis))
    project.mark_stage("source", key)
    progress("source", 1.0, "Recording decoded")


def _check(cancel: Callable[[], bool]) -> None:
    if cancel():
        raise Cancelled()


# --- stage 2: separation passes ------------------------------------------------


def pass_key(spec: PassSpec, settings: Settings, input_key: str) -> str:
    m, h = settings.models, settings.hardware
    return _key(spec.name, spec.model, input_key, m.roformer_overlap, m.demucs_shifts, m.vocal_ensemble_algorithm,
                h.chunk_seconds, h.chunk_overlap_seconds, h.half_precision)


def chunk_starts(total: int, chunk: int, overlap: int) -> list[int]:
    """Start frames of overlapping chunks covering ``total`` frames."""
    if total <= chunk:
        return [0]
    step = chunk - overlap
    starts = [0]
    while starts[-1] + chunk < total:
        starts.append(starts[-1] + step)
    return starts


def run_separation(project: Project, backend: SeparationBackend, progress: ProgressFn = _noop_progress,
                   cancel: Callable[[], bool] = lambda: False) -> None:
    settings = project.settings
    passes = build_passes(settings.models)
    keys = {"source": source_key(project)}
    weights = [_pass_weight(p) for p in passes]
    done_weight = 0.0
    for spec, weight in zip(passes, weights):
        key = pass_key(spec, settings, keys[spec.input])
        for out in spec.outputs:
            keys[out] = key
        if project.stage_done(spec.name, key) and all((project.stems_dir / f"{o}.flac").is_file() for o in spec.outputs):
            done_weight += weight
            continue

        def pass_progress(fraction: float, message: str, _base=done_weight, _w=weight):
            progress("separation", (_base + _w * fraction) / sum(weights), message)

        _run_pass(project, spec, key, backend, settings, pass_progress, cancel)
        project.mark_stage(spec.name, key, model=spec.model)
        done_weight += weight
    project.update(lambda state: state.__setitem__("aliases", stem_aliases(passes)))
    progress("separation", 1.0, "All stems separated")


def _pass_weight(spec: PassSpec) -> float:
    """Rough relative cost, so the overall progress bar moves evenly."""
    models = spec.model.split("+")
    per_model = 1.0 if spec.model.endswith(".onnx") or spec.model.endswith(".pth") else 4.0
    return per_model * len(models)


def _run_pass(project: Project, spec: PassSpec, key: str, backend: SeparationBackend, settings: Settings,
              progress: Callable[[float, str], None], cancel: Callable[[], bool]) -> None:
    source = project.stems_dir / f"{spec.input}.flac"
    total = stem_frames(source)
    chunk = int(settings.hardware.chunk_seconds * SAMPLE_RATE)
    overlap = min(int(settings.hardware.chunk_overlap_seconds * SAMPLE_RATE), chunk // 4)
    starts = chunk_starts(total, chunk, overlap)
    chunk_dir = project.work_dir / "chunks" / spec.name
    marker_key = chunk_dir / "key.txt"
    if chunk_dir.exists() and (not marker_key.exists() or marker_key.read_text() != key):
        shutil.rmtree(chunk_dir)  # results from different settings
    chunk_dir.mkdir(parents=True, exist_ok=True)
    marker_key.write_text(key)

    for index, start in enumerate(starts):
        done_marker = chunk_dir / f"{index:04d}.done"
        if done_marker.exists():
            continue
        _check(cancel)
        stop = min(total, start + chunk)
        progress(index / len(starts), f"{spec.label}: part {index + 1} of {len(starts)}")
        audio = read_stem(source, start, stop)
        outputs = spec.split(backend.separate(audio, SAMPLE_RATE, spec.model), audio, spec.model)
        missing = set(spec.outputs) - set(outputs)
        if missing:
            raise RuntimeError(f"{spec.model} did not produce {', '.join(sorted(missing))}")
        for name in spec.outputs:
            with StemWriter(chunk_dir / f"{index:04d}_{name}.flac") as writer:
                writer.write(outputs[name])
        done_marker.write_text("ok")

    progress(0.99, f"{spec.label}: joining parts")
    for name in spec.outputs:
        _assemble(chunk_dir, name, len(starts), overlap, project.stems_dir / f"{name}.flac")
    shutil.rmtree(chunk_dir, ignore_errors=True)
    progress(1.0, f"{spec.label}: done")


def _assemble(chunk_dir: Path, name: str, count: int, overlap: int, target: Path) -> None:
    """Join chunk results, crossfading the overlaps linearly (the chunks hold the same, correlated audio)."""
    fade_in = np.linspace(0.0, 1.0, overlap, dtype=np.float32)[None, :] if overlap else None
    with StemWriter(target) as writer:
        tail = None
        for index in range(count):
            data = read_stem(chunk_dir / f"{index:04d}_{name}.flac")
            if tail is not None:
                head = data[:, :overlap]
                writer.write(tail * (1.0 - fade_in[:, : head.shape[-1]]) + head * fade_in[:, : head.shape[-1]])
                data = data[:, overlap:]
            if index < count - 1 and overlap:
                writer.write(data[:, :-overlap])
                tail = data[:, -overlap:]
            else:
                writer.write(data)
