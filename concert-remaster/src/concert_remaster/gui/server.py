"""The app's local server: a small web API plus the page that talks to it.

Everything runs on this PC (127.0.0.1 only). Heavy jobs run in a separate
worker process so the window stays responsive, a crash can't take the app
down, and Stop is instant: the engine caches every finished chunk, so a
stopped job continues where it left off.
"""

from __future__ import annotations

import hashlib
import io
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
    from fastapi.responses import FileResponse, JSONResponse, Response
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
                "settings": project.state.get("settings"), "render": project.state.get("render"),
                "aliases": project.state.get("aliases", {}), "outputs": list_outputs(project)}

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
        if task not in ("analyze", "redetect", "identify", "export", "all"):
            raise HTTPException(400, "Unknown task")
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
    def peaks(project_id: str):
        project = get(project_id)
        if not project.has_stem("source"):
            return {"rate": 0, "peaks": []}
        return {"rate": PEAKS_PER_SECOND, "peaks": waveform_peaks(project).round(3).tolist()}

    @app.get("/api/projects/{project_id}/audio")
    def audio(project_id: str, stem: str = "source", start: float = 0.0, end: float = 30.0):
        project = get(project_id)
        end = min(end, start + 600.0)
        names = [n for n in stem.split("+") if n]
        a, b = int(start * SAMPLE_RATE), int(end * SAMPLE_RATE)
        parts = [read_stem(project.stem_path(n), a, b) for n in names if project.has_stem(n)]
        if not parts:
            raise HTTPException(404, "Stem not available yet")
        return Response(wav_bytes(np.sum(parts, axis=0)), media_type="audio/wav")

    @app.post("/api/projects/{project_id}/preview")
    def preview(project_id: str, data: dict = Body(...)):
        project = get(project_id)
        if data.get("settings"):
            project.set_settings(settings_mod.from_dict(data["settings"]))
        audio_data = render_preview(project, data["segment"], float(data.get("start", 0)), float(data.get("seconds", 30)))
        return Response(wav_bytes(audio_data), media_type="audio/wav")

    @app.post("/api/projects/{project_id}/reference")
    def set_reference(project_id: str, data: dict = Body(...)):
        project = get(project_id)
        return set_manual_reference(project, data)

    @app.post("/api/open")
    def open_path(data: dict = Body(...)):
        path = Path(data["path"])
        if not path.exists():
            raise HTTPException(404, "Not found")
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


def wav_bytes(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> bytes:
    import soundfile as sf

    peak = float(np.abs(audio).max()) if audio.size else 0.0
    if peak > 0.99:
        audio = audio * (0.99 / peak)
    buffer = io.BytesIO()
    sf.write(buffer, np.asarray(audio, dtype=np.float32).T, sample_rate, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


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
    audio, _, _ = _render_song(project, seg, mix_stems(project.state["aliases"]), settings, guide,
                               _restore_cutoff(project, settings), a, b)
    return audio


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


def run(host: str = "127.0.0.1", port: int = 8765, open_window: bool = True) -> None:
    import uvicorn

    ensure_ffmpeg()
    projects_dir().mkdir(parents=True, exist_ok=True)
    url = f"http://{host}:{port}/"
    if open_window:
        threading.Timer(1.2, open_app_window, args=(url,)).start()
    print(f"Concert Remaster is running at {url}  (close this window to quit)")
    uvicorn.run(create_app(), host=host, port=port, log_level="warning")
