"""The app's local server: a small web API plus the page that talks to it.

Everything runs on this PC (127.0.0.1 only). Heavy jobs run in a separate
worker process so the window stays responsive, a crash can't take the app
down, and Stop is instant: the engine caches every finished chunk, so a
stopped job continues where it left off.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path
from urllib.parse import quote

import numpy as np

from .. import __version__
from .. import settings as settings_mod
from ..audio_io import SAMPLE_RATE, ensure_ffmpeg, read_stem
from ..engine import Project
from ..paths import app_root, projects_dir
from ..workflow import STAGE_LABELS, renumber

log = logging.getLogger(__name__)
STATIC = Path(__file__).parent / "static"
DEFAULTS_FILE = app_root() / "settings.json"


# --- helpers -------------------------------------------------------------------------


def default_settings() -> settings_mod.Settings:
    if DEFAULTS_FILE.exists():
        try:
            return settings_mod.load(DEFAULTS_FILE)
        except Exception:
            log.warning("Ignoring unreadable %s", DEFAULTS_FILE)
    return settings_mod.Settings()


def project_dirs() -> list[Path]:
    root = projects_dir()
    if not root.exists():
        return []
    return sorted((p for p in root.iterdir() if (p / Project.FILE).exists()), key=lambda p: p.stat().st_mtime, reverse=True)


def open_project(project_id: str) -> Project:
    path = (projects_dir() / project_id).resolve()
    if projects_dir().resolve() not in path.parents or not (path / Project.FILE).exists():
        raise KeyError(project_id)
    return Project(path)


def project_summary(project: Project, jobs: "JobManager") -> dict:
    state = project.state
    jobs.check_finished(project)
    progress = project.read_progress()
    running = jobs.running_for(project.root.name)
    if not running and progress.get("status") == "running":
        progress = {**progress, "status": "interrupted", "message": "Stopped before finishing. Start again to continue."}
    stages = state.get("stages", {})
    segments = state.get("segments", [])
    return {
        "id": project.root.name,
        "name": state["name"],
        "source": state["source"],
        "created": state.get("created"),
        "source_info": state.get("source_info", {}),
        "analysis": state.get("analysis", {}),
        "stages": list(stages),
        "analyzed": bool(segments),
        "songs": sum(1 for s in segments if s["kind"] == "song"),
        "exported": bool(state.get("render")),
        "progress": progress,
        "running": running,
    }


# --- jobs ------------------------------------------------------------------------------


class JobManager:
    """One heavy job at a time (the GPU can only do one thing well)."""

    def __init__(self):
        self.proc: subprocess.Popen | None = None
        self.project_id: str | None = None
        self.task: str | None = None
        self.lock = threading.Lock()

    def running_for(self, project_id: str) -> bool:
        return self.project_id == project_id and self.is_running()

    def check_finished(self, project: Project) -> None:
        """If the worker died without reporting (e.g. a driver error at start-up), show why."""
        if self.project_id != project.root.name or self.proc is None or self.proc.poll() is None:
            return
        progress = project.read_progress()
        if self.proc.returncode not in (0, 3) and progress.get("status") in ("running", None):
            logs = sorted((project.root / "logs").glob("*.log"))
            tail = logs[-1].read_text(encoding="utf-8", errors="replace").strip().splitlines()[-12:] if logs else []
            project.write_progress(status="error", task=self.task, message="\n".join(tail) or f"Worker exited with code {self.proc.returncode}")

    def is_running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def start(self, project: Project, task: str, only: list[str] | None = None) -> None:
        with self.lock:
            if self.is_running():
                raise RuntimeError("Another job is running. Wait for it or stop it first.")
            logs = project.root / "logs"
            logs.mkdir(exist_ok=True)
            log_file = open(logs / f"{time.strftime('%Y%m%d-%H%M%S')}-{task}.log", "w", encoding="utf-8")
            cmd = [sys.executable, "-m", "concert_remaster", "worker", str(project.root), task]
            if only:
                cmd += ["--only", *only]
            flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
            project.write_progress(status="running", stage="starting", fraction=0.0, overall=0.0, task=task,
                                   stage_label="Starting", message="Loading the AI engine")
            self.proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT, creationflags=flags)
            self.project_id, self.task = project.root.name, task

    def stop(self, project: Project) -> None:
        with self.lock:
            if self.running_for(project.root.name):
                (project.root / "cancel.request").write_text("stop")
                self.proc.terminate()
                try:
                    self.proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self.proc.kill()
                project.write_progress(status="cancelled", task=self.task, message="Stopped. Start again to continue where it left off.")


# --- app ---------------------------------------------------------------------------------


def create_app():
    from fastapi import Body, FastAPI, HTTPException
    from fastapi.responses import FileResponse, JSONResponse
    from fastapi.staticfiles import StaticFiles

    app = FastAPI(title="Concert Remaster", docs_url=None, redoc_url=None)
    jobs = JobManager()

    def get(project_id: str) -> Project:
        try:
            return open_project(project_id)
        except KeyError:
            raise HTTPException(404, "Project not found")

    @app.get("/api/info")
    def info():
        from ..cli import describe_devices

        return {"version": __version__, "devices": describe_devices(), "projects_dir": str(projects_dir()),
                "ffmpeg": bool(ensure_ffmpeg()), "stage_labels": STAGE_LABELS}

    @app.get("/api/ping")
    def ping():
        return {"app": "concert-remaster", "version": __version__}

    @app.post("/api/focus")
    def focus():
        """A second launch asks the open window to come to the front."""
        callback = getattr(app.state, "focus", None)
        if callback:
            callback()
        return {"ok": bool(callback)}

    @app.get("/api/schema")
    def schema():
        return settings_mod.schema()

    @app.get("/api/defaults")
    def defaults():
        return settings_mod.to_dict(default_settings())

    @app.put("/api/defaults")
    def save_defaults(data: dict = Body(...)):
        settings_mod.save(settings_mod.from_dict(data), DEFAULTS_FILE)
        return {"ok": True}

    @app.post("/api/preset")
    def preset(data: dict = Body(...)):
        s = settings_mod.from_dict(data.get("settings"))
        return settings_mod.to_dict(settings_mod.apply_preset(s, data["preset"]))

    @app.post("/api/style")
    def style(data: dict = Body(...)):
        s = settings_mod.from_dict(data.get("settings"))
        return settings_mod.to_dict(settings_mod.apply_style(s, data["style"]))

    @app.post("/api/browse")
    def browse(data: dict = Body(default={})):
        return {"path": native_dialog(data.get("kind", "file"), data.get("title", ""))}

    @app.get("/api/projects")
    def projects():
        out = []
        for path in project_dirs():
            try:
                out.append(project_summary(Project(path), jobs))
            except Exception as exc:
                log.warning("Skipping %s: %s", path, exc)
        return out

    @app.post("/api/projects")
    def create(data: dict = Body(...)):
        source = Path(data["source"].strip().strip('"'))
        if not source.is_file():
            raise HTTPException(400, f"File not found: {source}")
        settings = settings_mod.from_dict(data["settings"]) if data.get("settings") else default_settings()
        if data.get("preset"):
            settings = settings_mod.apply_preset(settings, data["preset"])
        project = Project.create(source, projects_dir(), settings, name=data.get("name") or None)
        if data.get("start", True):
            jobs.start(project, "analyze")
        return project_summary(project, jobs)

    @app.get("/api/projects/{project_id}")
    def project_detail(project_id: str):
        project = get(project_id)
        return {**project_summary(project, jobs), "segments": project.state.get("segments", []),
                "settings": settings_mod.to_dict(project.settings), "render": project.state.get("render"),
                "aliases": project.state.get("aliases", {}), "outputs": list_outputs(project),
                "effects": project.state.get("effects", [])}

    @app.delete("/api/projects/{project_id}")
    def delete(project_id: str, keep_outputs: bool = True):
        project = get(project_id)
        if jobs.running_for(project_id):
            raise HTTPException(409, "Stop the running job first")
        if keep_outputs:
            shutil.rmtree(project.work_dir, ignore_errors=True)
            return {"ok": True, "kept": str(project.output_dir)}
        shutil.rmtree(project.root, ignore_errors=True)
        return {"ok": True}

    @app.put("/api/projects/{project_id}/settings")
    def put_settings(project_id: str, data: dict = Body(...)):
        project = get(project_id)
        project.set_settings(settings_mod.from_dict(data))
        return {"ok": True}

    @app.put("/api/projects/{project_id}/segments")
    def put_segments(project_id: str, data: list = Body(...)):
        project = get(project_id)
        duration = project.duration
        segments = sorted((dict(s) for s in data), key=lambda s: s["start"])
        for s in segments:
            s["start"] = max(0.0, min(float(s["start"]), duration))
            s["end"] = max(s["start"], min(float(s["end"]), duration))
            if s.get("kind") not in ("song", "interlude", "talk", "crowd", "silence"):
                raise HTTPException(400, f"Unknown segment kind {s.get('kind')!r}")
        segments = renumber(segments)
        project.update(lambda state: state.__setitem__("segments", segments))
        return segments

    @app.post("/api/projects/{project_id}/run")
    def run_task(project_id: str, data: dict = Body(...)):
        project = get(project_id)
        task = data.get("task", "analyze")
        if task not in ("analyze", "redetect", "identify", "studio", "export", "all"):
            raise HTTPException(400, "Unknown task")
        from ..playback import get_player

        get_player().release(project_id)  # the job may rewrite files that are playing
        try:
            jobs.start(project, task, data.get("only"))
        except RuntimeError as exc:
            raise HTTPException(409, str(exc))
        return {"ok": True}

    @app.post("/api/projects/{project_id}/stop")
    def stop(project_id: str):
        jobs.stop(get(project_id))
        return {"ok": True}

    @app.get("/api/projects/{project_id}/progress")
    def progress(project_id: str):
        project = get(project_id)
        return {**project.read_progress(), "running": jobs.running_for(project_id)}

    @app.get("/api/projects/{project_id}/peaks")
    def peaks(project_id: str, stem: str = "", start: float = 0.0, end: float = 0.0, rate: int = 20,
              source: str = "raw", seg: str = ""):
        project = get(project_id)
        if not stem:
            if not project.has_stem("source"):
                return {"rate": 0, "peaks": []}
            return {"rate": PEAKS_PER_SECOND, "peaks": waveform_peaks(project).round(3).tolist()}
        path, offset = track_file(project, stem, source, seg)
        return {"rate": rate, "peaks": range_peaks(path, start - offset, end - offset, rate)}

    @app.get("/api/projects/{project_id}/tracks/{seg_id}")
    def tracks(project_id: str, seg_id: str):
        return mixer_tracks(get(project_id), seg_id)

    @app.post("/api/projects/{project_id}/preview")
    def preview(project_id: str, data: dict = Body(...)):
        """Render a stretch of a song with the full export chain and play it."""
        from ..playback import ArrayTrack, Channel, Session

        project = get(project_id)
        if data.get("settings"):
            project.set_settings(settings_mod.from_dict(data["settings"]))
        seg = next(s for s in project.state.get("segments", []) if s["id"] == data["segment"])
        start = seg["start"] + float(data.get("start", 0))
        audio_data = render_preview(project, data["segment"], float(data.get("start", 0)), float(data.get("seconds", 30)))
        session = Session("preview", start, start + audio_data.shape[-1] / SAMPLE_RATE,
                          {"export": Channel(ArrayTrack(audio_data, start))},
                          {"project": project_id, "seg": seg["id"], "label": f"Export preview · {seg.get('title') or 'song'}"})
        return player_load(session, master=1.0)

    @app.get("/api/player")
    def player_status():
        from ..playback import get_player

        return get_player().status()

    @app.post("/api/player")
    def player_command(data: dict = Body(...)):
        return player_control(data, get)

    @app.post("/api/projects/{project_id}/reference")
    def set_reference(project_id: str, data: dict = Body(...)):
        project = get(project_id)
        return set_manual_reference(project, data)

    @app.post("/api/open")
    def open_path(data: dict = Body(...)):
        path = Path(data["path"])
        if not path.exists():
            raise HTTPException(404, "Not found")
        if data.get("launch") and path.is_file():
            launch(path)  # e.g. a video, in the PC's own player
        else:
            reveal(path)
        return {"ok": True}

    @app.get("/files/{project_id}/{relative:path}")
    def files(project_id: str, relative: str):
        project = get(project_id)
        path = (project.output_dir / relative).resolve()
        if project.output_dir.resolve() not in path.parents or not path.is_file():
            raise HTTPException(404, "Not found")
        return FileResponse(path)

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-store"})

    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    @app.exception_handler(Exception)
    async def errors(request, exc):
        log.exception("Request failed")
        return JSONResponse({"detail": f"{type(exc).__name__}: {exc}"}, status_code=500)

    app.state.jobs = jobs
    return app


# --- feature helpers ---------------------------------------------------------------------

PEAKS_PER_SECOND = 10


def player_load(session, master: float | None = None, position: float | None = None, play: bool | None = True,
                loop=None) -> dict:
    from fastapi import HTTPException

    from ..playback import AudioUnavailable, get_player

    player = get_player()
    player.set_loop(*(loop or (False,)))
    try:
        player.load(session, position=position, play=play)
        if master is not None:
            player.set_mix(master=master)
    except AudioUnavailable as exc:
        raise HTTPException(503, str(exc))
    return player.status()


def player_control(data: dict, get) -> dict:
    """Commands from the window to the audio engine."""
    import soundfile as sf
    from fastapi import HTTPException

    from ..playback import AudioUnavailable, Channel, FileTrack, Session, get_player

    player = get_player()
    cmd = data.get("cmd")
    if cmd == "song":
        project = get(data["project"])
        session = song_session(project, data["seg"], data.get("source", "raw"), data.get("gains") or {}, data.get("pans") or {})
        loop = (bool(data.get("loop")), *(data.get("loop_range") or (None, None)))
        return player_load(session, master=data.get("master", 1.0), position=data.get("position"),
                           play=data.get("play"), loop=loop)
    if cmd == "clip":
        project = get(data["project"])
        stems = [n for n in data.get("stems") or ["source"] if project.has_stem(n)]
        if not stems:
            raise HTTPException(404, "Nothing to play yet")
        start = max(0.0, float(data.get("start", 0)))
        end = min(project.duration, float(data.get("end", start + 30)))
        session = Session("clip", start, end, {n: Channel(FileTrack(project.stem_path(n))) for n in stems},
                          {"project": data["project"], "label": data.get("label") or "Original recording"})
        return player_load(session, master=1.0)
    if cmd == "file":
        project = get(data["project"])
        path = (project.output_dir / data["path"]).resolve()
        if project.output_dir.resolve() not in path.parents or not path.is_file():
            raise HTTPException(404, "Not found")
        duration = sf.info(str(path)).duration
        session = Session("file", 0.0, duration, {"file": Channel(FileTrack(path, stored_stem=False))},
                          {"project": data["project"], "label": path.name})
        return player_load(session, master=1.0)
    try:
        if cmd == "play":
            player.play(data.get("position"))
        elif cmd == "pause":
            player.pause()
        elif cmd == "stop":
            player.stop()
        elif cmd == "seek":
            player.seek(float(data["position"]))
        elif cmd == "mix":
            player.set_mix(data.get("gains"), data.get("pans"), data.get("master"))
        elif cmd == "loop":
            player.set_loop(bool(data.get("on")), data.get("start"), data.get("end"))
        else:
            raise HTTPException(400, f"Unknown player command {cmd!r}")
    except AudioUnavailable as exc:
        raise HTTPException(503, str(exc))
    return player.status()


def song_session(project: Project, seg_id: str, source: str, gains: dict, pans: dict):
    """A song's tracks for the mixer: the original, the raw AI stems or the studio tracks."""
    from fastapi import HTTPException

    from .. import studio
    from ..playback import Channel, FileTrack, Session
    from ..separation import mix_stems

    seg = next((s for s in project.state.get("segments", []) if s["id"] == seg_id), None)
    if seg is None:
        raise HTTPException(404, "No such part")
    if source == "original":
        files = {"source": (project.stem_path("source"), 0.0)}
    elif source == "studio":
        meta = studio.read_meta(project, seg_id)
        if meta is None:
            raise HTTPException(404, "Studio tracks are not prepared for this song")
        files = {n: (studio.track_path(project, seg_id, n), float(meta["start"])) for n in meta["tracks"]}
    else:
        names = mix_stems(project.state.get("aliases", {}), project.settings.mix.drum_kit)
        names += ["crowd"] if project.has_stem("crowd") else []
        files = {n: (project.stem_path(n), 0.0) for n in names if project.has_stem(n)}
    channels = {n: Channel(FileTrack(path, offset), gain=float(gains.get(n, 1.0)), pan=float(pans.get(n, 0.0)))
                for n, (path, offset) in files.items() if path.is_file()}
    end = min(seg["end"], project.duration)
    return Session("song", seg["start"], end, channels,
                   {"project": project.root.name, "seg": seg_id, "source": source,
                    "label": f"{seg.get('track') or ''} {seg.get('title') or 'Song'}".strip()})


def track_file(project: Project, stem: str, source: str, seg_id: str) -> tuple[Path, float]:
    """The file behind a mixer track, and the show time its first frame belongs to."""
    from fastapi import HTTPException

    from .. import studio

    if source == "studio":
        meta = studio.read_meta(project, seg_id)
        path = studio.track_path(project, seg_id, stem)
        if meta is None or not path.is_file():
            raise HTTPException(404, "Studio tracks are not prepared for this song")
        return path, float(meta["start"])
    if not project.has_stem(stem):
        raise HTTPException(404, f"No {stem} track")
    return project.stem_path(stem), 0.0


_PEAK_CACHE: dict[tuple, list] = {}


def range_peaks(path: Path, start: float, end: float, rate: int) -> list[float]:
    """Peak level per 1/``rate`` s of a stretch of a track (not normalised: quiet tracks look quiet)."""
    rate = max(1, min(200, rate))
    key = (str(path), path.stat().st_mtime, round(start, 2), round(end, 2), rate)
    if key in _PEAK_CACHE:
        return _PEAK_CACHE[key]
    a, b = max(0, int(start * SAMPLE_RATE)), max(0, int(end * SAMPLE_RATE))
    hop = SAMPLE_RATE // rate
    out: list[np.ndarray] = []
    for block_start in range(a, b, hop * 3000):
        block = read_stem(path, block_start, min(b, block_start + hop * 3000))
        mono = np.abs(block).max(axis=0) if block.size else np.zeros(0)
        n = -(-mono.size // hop)
        if n:
            mono = np.pad(mono, (0, n * hop - mono.size))
            out.append(mono.reshape(n, hop).max(axis=1))
    peaks = np.round(np.concatenate(out), 3).tolist() if out else []
    if len(_PEAK_CACHE) > 256:
        _PEAK_CACHE.clear()
    _PEAK_CACHE[key] = peaks
    return peaks


def mixer_tracks(project: Project, seg_id: str) -> dict:
    """Everything the app's mixer needs for one song: tracks, fader states, stage effects."""
    from fastapi import HTTPException

    from .. import studio
    from ..render import track_state
    from ..separation import mix_stems

    seg = next((s for s in project.state.get("segments", []) if s["id"] == seg_id), None)
    if seg is None:
        raise HTTPException(404, "No such part")
    settings = project.settings
    raw = mix_stems(project.state.get("aliases", {}), settings.mix.drum_kit)
    raw += ["crowd"] if project.has_stem("crowd") else []
    state = studio.status(project, seg, settings)
    meta = studio.read_meta(project, seg_id) or {}
    names = list(dict.fromkeys(raw + meta.get("tracks", [])))
    effects = [e for e in project.state.get("effects") or [] if e["end"] > seg["start"] and e["start"] < seg["end"]]
    return {
        "segment": seg_id,
        "start": seg["start"],
        "end": seg["end"],
        "raw_tracks": raw,
        "studio_tracks": meta.get("tracks", []),
        "studio": state,
        "studio_key": meta.get("key"),
        "tracks": {n: {**track_state(n, seg, settings), "auto_gain_db": meta.get("auto_gains_db", {}).get(n, 0.0),
                       "empty": n in meta.get("empty", [])} for n in names},
        "monitor_gain_db": {"raw": studio.raw_monitor_gain_db(project, seg, [n for n in raw if n != "crowd"], settings),
                            "studio": meta.get("monitor_gain_db", 0.0)},
        "effects": effects,
        "fx_action": seg.get("fx_action") or "",
        "default_fx_action": settings.effects.action,
    }


def waveform_peaks(project: Project) -> np.ndarray:
    """Peak level per 0.1 s of the whole show, cached."""
    cache = project.work_dir / "analysis" / "peaks.npy"
    source = project.stem_path("source")
    if cache.exists() and cache.stat().st_mtime >= source.stat().st_mtime:
        return np.load(cache)
    hop = SAMPLE_RATE // PEAKS_PER_SECOND
    out = []
    frames = project.frames
    for start in range(0, frames, hop * 600):
        block = read_stem(source, start, min(frames, start + hop * 600))
        mono = np.abs(block).max(axis=0)
        n = mono.size // hop
        if n:
            out.append(mono[: n * hop].reshape(n, hop).max(axis=1))
    peaks = np.concatenate(out) if out else np.zeros(0)
    peaks = peaks / max(float(peaks.max()) if peaks.size else 1.0, 1e-9)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, peaks.astype(np.float32))
    return peaks


def render_preview(project: Project, segment_id: str, start: float, seconds: float) -> np.ndarray:
    """Render a short stretch of one song with the current settings, for A/B listening."""
    from ..reference import load_guide
    from ..render import _render_song, _restore_cutoff
    from ..separation import mix_stems

    seg = next(s for s in project.state.get("segments", []) if s["id"] == segment_id)
    settings = project.settings
    a = int(max(seg["start"], seg["start"] + start) * SAMPLE_RATE)
    b = int(min(seg["end"], seg["start"] + start + seconds) * SAMPLE_RATE)
    guide = None
    ref = seg.get("reference")
    if ref and settings.reference.tone_match and Path(ref.get("path", "")).exists():
        guide = load_guide(ref, None, settings.models.instrument_model, False)  # cached stems are used if present
    audio, _, _ = _render_song(project, seg, mix_stems(project.state["aliases"], settings.mix.drum_kit), settings, guide,
                               _restore_cutoff(project, settings), a, b)
    return audio if audio is not None else np.zeros((2, max(0, b - a)), dtype=np.float32)


def set_manual_reference(project: Project, data: dict) -> dict:
    """Let the user name a song and point at its original (file or URL) by hand."""
    from dataclasses import asdict

    from ..identify import Candidate, Reference, ReferenceLibrary, download

    segment_id = data["segment"]
    seg = next(s for s in project.state.get("segments", []) if s["id"] == segment_id)
    library = ReferenceLibrary()
    ref = None
    if data.get("file"):
        path = Path(data["file"])
        ref = Reference("manual-" + hashlib.sha1(str(path).encode()).hexdigest()[:16], data.get("title") or path.stem, data.get("artist", ""), str(path), "manual")
    elif data.get("url"):
        url = data["url"].strip()
        source = "youtube" if "youtu" in url else "itunes" if "apple.com" in url else "deezer" if "dzcdn" in url else "web"
        cand = Candidate(data.get("title", ""), data.get("artist", ""), source if source != "web" else "youtube", url)
        path = download(cand, library.root / "downloads", project.settings.identify.cookies_browser)
        ref = Reference("manual-" + hashlib.sha1(url.encode()).hexdigest()[:16], data.get("title") or path.stem, data.get("artist", ""), str(path), "manual", url=url)
    if ref is not None:
        library.add(ref)
    seg.update({"title": data.get("title") or seg.get("title"), "artist": data.get("artist", seg.get("artist", "")),
                "reference": asdict(ref) if ref else (None if data.get("clear") else seg.get("reference")), "locked": True})
    project.update(lambda state: state.__setitem__("segments", [seg if s["id"] == segment_id else s for s in state["segments"]]))
    return seg


def list_outputs(project: Project) -> list[dict]:
    root = project.output_dir
    if not root.exists():
        return []
    files = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and ".partial" not in path.name:
            rel = path.relative_to(root).as_posix()
            files.append({"path": rel, "name": path.name, "folder": path.parent.relative_to(root).as_posix(),
                          "size": path.stat().st_size, "url": f"/files/{quote(project.root.name)}/{quote(rel)}",
                          "full_path": str(path)})
    return files


def native_dialog(kind: str, title: str) -> str:
    """A real Windows/macOS/Linux file picker, run in its own process (Tk must own its thread)."""
    code = (
        "import tkinter as tk, tkinter.filedialog as fd, sys\n"
        "r = tk.Tk(); r.withdraw(); r.attributes('-topmost', True)\n"
        f"kind = {kind!r}\n"
        "types = [('Recordings', '*.mp3 *.mp4 *.m4a *.wav *.flac *.aac *.mov *.mkv *.webm *.ogg *.opus *.3gp *.amr'), ('All files', '*.*')]\n"
        f"p = fd.askdirectory(title={title or 'Choose a folder'!r}) if kind == 'folder' else fd.askopenfilename(title={title or 'Choose a recording'!r}, filetypes=types)\n"
        "sys.stdout.write(p or '')\n"
    )
    try:
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=600,
                                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        return result.stdout.strip()
    except Exception as exc:
        log.warning("File dialog failed: %s", exc)
        return ""


def launch(path: Path) -> None:
    if os.name == "nt":
        os.startfile(str(path))  # type: ignore[attr-defined]
    else:
        subprocess.Popen(["open" if sys.platform == "darwin" else "xdg-open", str(path)])


def reveal(path: Path) -> None:
    if os.name == "nt":
        if path.is_file():
            subprocess.Popen(["explorer", "/select,", str(path)])
        else:
            os.startfile(str(path))  # type: ignore[attr-defined]
    elif sys.platform == "darwin":
        subprocess.Popen(["open", "-R" if path.is_file() else "", str(path)])
    else:
        subprocess.Popen(["xdg-open", str(path if path.is_dir() else path.parent)])


def open_app_window(url: str) -> None:
    """Open as a clean app window (Edge app mode on Windows), or a browser tab elsewhere."""
    if os.name == "nt":
        for base in (os.environ.get("PROGRAMFILES(X86)"), os.environ.get("PROGRAMFILES"), os.environ.get("LOCALAPPDATA")):
            edge = Path(base or "") / "Microsoft" / "Edge" / "Application" / "msedge.exe"
            if base and edge.exists():
                subprocess.Popen([str(edge), f"--app={url}", "--window-size=1500,950"])
                return
    webbrowser.open(url)


def desktop_window(url: str, app=None) -> bool:
    """Show the app in its own desktop window (WebView2 on Windows). Returns when it is closed."""
    try:
        import webview
    except ImportError:
        return False
    try:
        window = webview.create_window("Concert Remaster", url, width=1560, height=980, min_size=(1100, 720),
                                       background_color="#0d0f14", text_select=True)

        def focus():
            window.restore()
            window.show()
            window.on_top = True
            window.on_top = False

        if app is not None:
            app.state.focus = focus
        webview.start(gui="edgechromium" if os.name == "nt" else None, private_mode=False,
                      storage_path=str(app_root() / "window"))
        return True
    except Exception as exc:
        log.warning("Desktop window unavailable (%s); opening an app window instead", exc)
        return False


def _running_instance(host: str, port: int) -> bool:
    """Is Concert Remaster already running on this port? Then bring its window forward."""
    import urllib.request

    try:
        with urllib.request.urlopen(f"http://{host}:{port}/api/ping", timeout=1.5) as r:
            if json.loads(r.read()).get("app") != "concert-remaster":
                return False
        urllib.request.urlopen(urllib.request.Request(f"http://{host}:{port}/api/focus", data=b"", method="POST"), timeout=3)
        return True
    except Exception:
        return False


def _free_port(host: str, port: int) -> int:
    import socket

    for candidate in range(port, port + 20):
        with socket.socket() as sock:
            try:
                sock.bind((host, candidate))
                return candidate
            except OSError:
                continue
    raise SystemExit("No free port for the app's local engine.")


def run(host: str = "127.0.0.1", port: int = 8765, open_window: bool = True) -> None:
    """Start the local engine (reachable only from this PC) and open the app window."""
    import uvicorn

    from ..playback import get_player

    if open_window and _running_instance(host, port):
        return  # already open: its window was brought to the front
    ensure_ffmpeg()
    projects_dir().mkdir(parents=True, exist_ok=True)
    port = _free_port(host, port)
    url = f"http://{host}:{port}/"
    app = create_app()
    # log_config=None: no console logging setup (there is no console when started from the app icon).
    server = uvicorn.Server(uvicorn.Config(app, host=host, port=port, log_level="warning", log_config=None, access_log=False))
    if not open_window:
        print(f"Concert Remaster engine running at {url}")
        server.run()
        return
    thread = threading.Thread(target=server.run, daemon=True, name="engine")
    thread.start()
    for _ in range(300):
        if server.started or not thread.is_alive():
            break
        time.sleep(0.05)
    if not thread.is_alive():
        raise SystemExit(f"Could not start: port {port} is in use (is Concert Remaster already open?)")
    try:
        if not desktop_window(url, app):
            open_app_window(url)
            print("Concert Remaster is running. Close this window to quit.")
            while thread.is_alive():
                thread.join(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        jobs = app.state.jobs
        if jobs.project_id and jobs.is_running():
            # Stop the running job cleanly; it continues from where it left off next time.
            try:
                jobs.stop(open_project(jobs.project_id))
            except Exception:
                log.exception("Could not stop the running job")
        get_player().close()
        server.should_exit = True
        thread.join(5)
