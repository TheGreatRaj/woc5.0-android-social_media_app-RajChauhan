"""Command line: ``concert-remaster`` opens the app; subcommands run things headless.

    concert-remaster                      open the app (same as `gui`)
    concert-remaster process show.mp4     analyze and export without the app
    concert-remaster download-models      fetch every model for offline use
    concert-remaster devices              show which GPU will be used
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
import threading
import time
from pathlib import Path

from . import __version__
from . import settings as settings_mod
from .paths import models_dir, projects_dir


def _parse_value(text: str):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def _apply_overrides(settings, overrides: list[str]):
    """``--set master.target_lufs=-10 --set speech.action=remove``"""
    data = settings_mod.to_dict(settings)
    for item in overrides:
        key, sep, value = item.partition("=")
        if not sep or "." not in key:
            raise SystemExit(f"--set expects group.name=value, got {item!r}")
        group, name = key.split(".", 1)
        if group not in data or name not in data[group]:
            raise SystemExit(f"Unknown setting {key!r}")
        data[group][name] = _parse_value(value)
    return settings_mod.from_dict(data)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="concert-remaster", description="Studio-quality remasters of live concert recordings, on your own PC.")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command")

    gui = sub.add_parser("gui", help="open the app (default)")
    gui.add_argument("--port", type=int, default=8765)
    gui.add_argument("--no-browser", action="store_true", help="don't open a window, just serve")
    gui.add_argument("--host", default="127.0.0.1")

    process = sub.add_parser("process", help="analyze and export recordings without the app")
    process.add_argument("inputs", nargs="+", type=Path)
    process.add_argument("-o", "--projects", type=Path, default=None, help=f"where project folders go (default {projects_dir()})")
    process.add_argument("--preset", choices=sorted(settings_mod.PRESETS), default="ultra")
    process.add_argument("--settings", type=Path, help="a settings JSON file saved from the app")
    process.add_argument("--set", action="append", default=[], metavar="GROUP.NAME=VALUE", help="change any setting, e.g. --set speech.action=remove")

    models = sub.add_parser("download-models", help="download every model for offline use")
    models.add_argument("--preset", choices=[*sorted(settings_mod.PRESETS), "all"], default="all")
    models.add_argument("--whisper", default="large-v3", help="Whisper model size to fetch ('' to skip)")

    worker = sub.add_parser("worker", help=argparse.SUPPRESS)
    worker.add_argument("project", type=Path)
    worker.add_argument("task", choices=["analyze", "redetect", "identify", "studio", "export", "all"])
    worker.add_argument("--only", nargs="*", help="segment ids for identify / studio")

    sub.add_parser("presets", help="list quality presets and their models")
    devices = sub.add_parser("devices", help="show the processing devices that will be used")
    devices.add_argument("--require", choices=["cuda"], help="exit 3 if PyTorch has no CUDA build, 4 if CUDA can't use the GPU")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    command = args.command or "gui"
    if command == "gui":
        _log_to_file()
        from .gui.server import run

        run(host=getattr(args, "host", "127.0.0.1"), port=getattr(args, "port", 8765), open_window=not getattr(args, "no_browser", False))
        return 0
    if command == "presets":
        from .separation import all_models

        for name, preset in settings_mod.PRESETS.items():
            s = settings_mod.apply_preset(settings_mod.Settings(), name)
            print(f"{name}: {preset['description']}")
            for model in all_models(s.models):
                print(f"    {model}")
        return 0
    if command == "devices":
        info = describe_devices()
        print(json.dumps(info, indent=2))
        if info.get("problem"):
            print(f"\n{info['problem']}\nFix: {info['fix']}")
        if getattr(args, "require", None) == "cuda" and not info["cuda"]:
            return 3 if not info.get("torch_cuda") else 4
        return 0
    if command == "download-models":
        return download_models(args.preset, args.whisper)
    if command == "worker":
        return run_worker(args.project, args.task, args.only)
    if command == "process":
        return process(args)
    parser.error(f"unknown command {command}")
    return 2


def _log_to_file() -> None:
    """The app usually runs without a console (started from its icon): keep a log file."""
    from logging.handlers import RotatingFileHandler

    from .paths import app_root

    try:
        folder = app_root() / "logs"
        folder.mkdir(parents=True, exist_ok=True)
        handler = RotatingFileHandler(folder / "app.log", maxBytes=2_000_000, backupCount=2, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        logging.getLogger().addHandler(handler)
    except OSError:
        pass


# PyTorch's CUDA 12.x builds need at least this NVIDIA driver on Windows (CUDA 12 minor-version compatibility).
MIN_NVIDIA_DRIVER = (527, 41)


def _nvidia_gpus() -> list[dict]:
    """NVIDIA GPUs and their driver version, from nvidia-smi or (Windows) the device list."""
    import shutil
    import subprocess

    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    smi = shutil.which("nvidia-smi") or (r"C:\Windows\System32\nvidia-smi.exe" if os.name == "nt" else None)
    if smi and os.path.exists(smi):
        try:
            out = subprocess.run([smi, "--query-gpu=name,driver_version", "--format=csv,noheader"], capture_output=True,
                                 text=True, timeout=20, creationflags=flags).stdout
            gpus = [dict(zip(("name", "driver"), (x.strip() for x in line.split(",", 1)))) for line in out.splitlines() if "," in line]
            if gpus:
                return gpus
        except Exception:
            pass
    if os.name == "nt":
        try:
            script = "Get-CimInstance Win32_VideoController | ForEach-Object { $_.Name + '|' + $_.DriverVersion }"
            out = subprocess.run(["powershell", "-NoProfile", "-Command", script], capture_output=True, text=True,
                                 timeout=30, creationflags=flags).stdout
            gpus = []
            for line in out.splitlines():
                name, _, version = line.strip().partition("|")
                if "NVIDIA" in name.upper():
                    gpus.append({"name": name, "driver": nvidia_driver_from_windows(version)})
            return gpus
        except Exception:
            pass
    return []


def nvidia_driver_from_windows(version: str) -> str:
    """Windows reports NVIDIA driver 552.22 as 31.0.15.5222: the last five digits are the driver."""
    digits = version.replace(".", "")[-5:]
    return f"{int(digits[:3])}.{digits[3:]}" if len(digits) == 5 and digits.isdigit() else version


def describe_devices() -> dict:
    """Which processor the AI will use, and if an NVIDIA GPU is left unused, why and how to fix it."""
    info: dict = {"cuda": False, "directml": False, "cpu_threads": os.cpu_count()}
    try:
        import torch

        info["torch"] = torch.__version__
        info["torch_cuda"] = torch.version.cuda  # None for a CPU-only build
        if torch.cuda.is_available():
            info["cuda"] = True
            info["gpu"] = torch.cuda.get_device_name(0)
            info["gpu_memory_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1)
        elif torch.version.cuda:
            try:
                torch.cuda.init()
            except Exception as exc:
                info["cuda_error"] = str(exc).strip().splitlines()[0][:300]
    except Exception as exc:
        info["torch_error"] = str(exc)
    try:
        import torch_directml

        if torch_directml.is_available():
            info["directml"] = True
            info["directml_device"] = torch_directml.device_name(0)
    except Exception:
        pass
    nvidia = [] if info["cuda"] else _nvidia_gpus()
    if nvidia:
        info["nvidia_gpu"], info["nvidia_driver"] = nvidia[0]["name"], nvidia[0]["driver"]
        info["problem"], info["fix"] = _why_no_cuda(info)
    info["recommended"] = "cuda" if info["cuda"] else "directml" if info["directml"] else "cpu"
    return info


def _why_no_cuda(info: dict) -> tuple[str, str]:
    gpu = info.get("nvidia_gpu", "the NVIDIA GPU")
    if "torch_error" in info:
        return f"The AI libraries could not start ({info['torch_error']}).", "Run setup.bat again to repair them."
    if not info.get("torch_cuda"):
        return (f"{gpu} is not used: the installed PyTorch ({info.get('torch')}) has no CUDA support, so the AI runs on the CPU.",
                "Run setup.bat again: it reinstalls the CUDA build of PyTorch (or run: setup.bat -Gpu nvidia).")
    try:
        driver = tuple(int(x) for x in str(info.get("nvidia_driver", "")).split(".")[:2])
    except ValueError:
        driver = ()
    if driver and driver < MIN_NVIDIA_DRIVER:
        return (f"{gpu} is not used: its driver ({info['nvidia_driver']}) is too old for CUDA {info['torch_cuda']}.",
                "Update the NVIDIA driver (GeForce Experience / NVIDIA App, or nvidia.com/drivers), restart, then start the app again.")
    detail = f" ({info['cuda_error']})" if info.get("cuda_error") else ""
    return (f"{gpu} is not available to CUDA{detail}.",
            "Update the NVIDIA driver and restart Windows. On laptops, make sure the NVIDIA GPU isn't disabled "
            "(Device Manager, or the laptop's power/graphics mode).")


def download_models(preset: str, whisper: str) -> int:
    from .audio_io import ensure_ffmpeg
    from .separation import all_models, model_file_ok

    ensure_ffmpeg()
    names: list[str] = []
    for name in (settings_mod.PRESETS if preset == "all" else [preset]):
        s = settings_mod.apply_preset(settings_mod.Settings(), name)
        names += all_models(s.models)
    names = list(dict.fromkeys(names))
    from audio_separator.separator import Separator

    target = models_dir()
    target.mkdir(parents=True, exist_ok=True)
    separator = Separator(info_only=True, model_file_dir=str(target), log_level=logging.ERROR)
    failures = 0
    for i, model in enumerate(names, 1):
        print(f"[{i}/{len(names)}] {model}", flush=True)
        path = target / model
        if path.exists() and not model_file_ok(path):
            print("    the earlier download is incomplete; fetching it again", flush=True)
            path.unlink()
        for attempt in range(3):
            try:
                with _heartbeat(target, f"    {model}"):
                    separator.download_model_and_data(model)
            except Exception as exc:
                error = str(exc)
            else:
                error = None if not path.exists() or model_file_ok(path) else "the download was incomplete"
            if error is None:
                break
            path.unlink(missing_ok=True)  # never leave a damaged file behind: it would be skipped next time
            print(f"    attempt {attempt + 1} failed: {error}", flush=True)
        else:
            failures += 1
            print(f"    FAILED: {model}", flush=True)
    # Weights fetched as part of other models (e.g. Demucs' .th files): drop damaged ones so a rerun fetches them.
    for weights in target.rglob("*"):
        if weights.suffix.lower() in (".ckpt", ".pth", ".th", ".onnx") and weights.is_file() and not model_file_ok(weights):
            print(f"    removing incomplete {weights.name}; run this again to fetch it", flush=True)
            weights.unlink()
            failures += 1
    if whisper:
        print(f"[whisper] {whisper}", flush=True)
        try:
            from faster_whisper.utils import download_model

            download_model(whisper, cache_dir=str(target / "whisper"))
        except Exception as exc:
            failures += 1
            print(f"    FAILED: {exc}", flush=True)
    print("All models are ready for offline use." if not failures else f"{failures} download(s) failed; run this again to retry.")
    return 1 if failures else 0


class _heartbeat:
    """Print a line every half minute during a long download (the installer shows these)."""

    def __init__(self, folder: Path, label: str):
        self.folder, self.label = folder, label
        self.done = threading.Event()

    def _size(self) -> int:
        return sum(f.stat().st_size for f in self.folder.rglob("*") if f.is_file())

    def __enter__(self):
        start, begin = time.time(), self._size()

        def beat():
            while not self.done.wait(30):
                got = (self._size() - begin) / 2**20
                print(f"{self.label}: {got:.0f} MB so far ({(time.time() - start) / 60:.0f} min)", flush=True)

        threading.Thread(target=beat, daemon=True).start()
        return self

    def __exit__(self, *exc):
        self.done.set()


def run_worker(project_dir: Path, task: str, only: list[str] | None) -> int:
    """Run one job for the app in its own process; progress goes to progress.json."""
    from .engine import Cancelled, Project
    from .workflow import Job

    project = Project(project_dir)
    if project.settings.hardware.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, lambda *_: stop.set())
    cancel_file = project.root / "cancel.request"
    cancel_file.unlink(missing_ok=True)

    def cancelled() -> bool:
        return stop.is_set() or cancel_file.exists()

    job = Job(project, cancel=cancelled)
    job.task = task
    try:
        if task == "analyze":
            job.analyze()
        elif task == "redetect":
            job.detect_segments(force=True)
            job.finish("Songs and speech detected again")
        elif task == "identify":
            job.identify(only=only)
            job.finish("Identification finished")
        elif task == "studio":
            job.prepare_studio(only=only)
            job.finish("Mixer tracks ready")
        elif task == "export":
            job.export()
        else:
            job.run_all()
        return 0
    except Cancelled:
        project.write_progress(status="cancelled", task=task, message="Stopped. Start again to continue where it left off.")
        return 3
    except Exception as exc:
        logging.exception("Job failed")
        project.write_progress(status="error", task=task, message=f"{type(exc).__name__}: {exc}")
        return 1
    finally:
        cancel_file.unlink(missing_ok=True)


def process(args) -> int:
    from .engine import Project
    from .workflow import Job

    settings = settings_mod.load(args.settings) if args.settings else settings_mod.apply_preset(settings_mod.Settings(), args.preset)
    settings = _apply_overrides(settings, args.set)
    failures = 0
    for path in args.inputs:
        print(f"== {path}")
        try:
            project = Project.create(path, args.projects, settings)
            last = {"t": 0.0}

            def show(info):
                if time.time() - last["t"] > 2 or info["fraction"] >= 1:
                    eta = f", about {info['eta_seconds'] // 60} min left" if info.get("eta_seconds") else ""
                    print(f"   [{info['overall']:6.1%}] {info['stage_label']}: {info['message']}{eta}", flush=True)
                    last["t"] = time.time()

            Job(project, on_progress=show).run_all()
            print(f"   Done: {project.output_dir}")
        except Exception as exc:
            failures += 1
            print(f"   FAILED: {exc}", file=sys.stderr)
            if args.verbose:
                raise
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
