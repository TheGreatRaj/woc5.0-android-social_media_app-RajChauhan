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
    worker.add_argument("task", choices=["analyze", "redetect", "identify", "export", "all"])
    worker.add_argument("--only", nargs="*", help="segment ids for identify")

    sub.add_parser("presets", help="list quality presets and their models")
    sub.add_parser("devices", help="show the processing devices that will be used")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    command = args.command or "gui"
    if command == "gui":
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
        print(json.dumps(describe_devices(), indent=2))
        return 0
    if command == "download-models":
        return download_models(args.preset, args.whisper)
    if command == "worker":
        return run_worker(args.project, args.task, args.only)
    if command == "process":
        return process(args)
    parser.error(f"unknown command {command}")
    return 2


def describe_devices() -> dict:
    info: dict = {"cuda": False, "directml": False, "cpu_threads": os.cpu_count()}
    try:
        import torch

        info["torch"] = torch.__version__
        if torch.cuda.is_available():
            info["cuda"] = True
            info["gpu"] = torch.cuda.get_device_name(0)
            info["gpu_memory_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1)
    except Exception as exc:
        info["torch_error"] = str(exc)
    try:
        import torch_directml

        if torch_directml.is_available():
            info["directml"] = True
            info["directml_device"] = torch_directml.device_name(0)
    except Exception:
        pass
    info["recommended"] = "cuda" if info["cuda"] else "directml" if info["directml"] else "cpu"
    return info


def download_models(preset: str, whisper: str) -> int:
    from .audio_io import ensure_ffmpeg
    from .separation import all_models, split_models

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
        try:
            separator.download_model_and_data(model)
        except Exception as exc:
            failures += 1
            print(f"    FAILED: {exc}", flush=True)
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
    try:
        if task == "analyze":
            job.analyze()
        elif task == "redetect":
            job.detect_segments(force=True)
            job.finish("Songs and speech detected again")
        elif task == "identify":
            job.identify(only=only)
            job.finish("Identification finished")
        elif task == "export":
            job.export()
        else:
            job.run_all()
        return 0
    except Cancelled:
        project.write_progress(status="cancelled", message="Stopped. Start again to continue where it left off.")
        return 3
    except Exception as exc:
        logging.exception("Job failed")
        project.write_progress(status="error", message=f"{type(exc).__name__}: {exc}")
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
