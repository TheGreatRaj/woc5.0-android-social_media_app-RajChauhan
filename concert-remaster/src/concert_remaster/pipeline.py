"""End-to-end remaster of one concert recording."""

from __future__ import annotations

import json
import logging
import math
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from .analysis import integrated_lufs
from .audio_io import SAMPLE_RATE, load_audio, save_audio
from .mastering import master
from .mixing import auto_balance, sum_stems
from .restoration import declip, remove_rumble
from .separation import PRESETS, AudioSeparatorBackend, SeparationBackend, build_plan, separate_concert
from .stem_processing import process_stem, profile_for

log = logging.getLogger(__name__)

DEFAULT_MODEL_DIR = Path.home() / ".cache" / "concert-remaster" / "models"


@dataclass
class RemasterSettings:
    preset: str = "balanced"
    stems: int = 6
    # Per-stage model overrides: crowd, vocals, instruments, dereverb, denoise ("none" disables).
    model_overrides: dict[str, str] = field(default_factory=dict)
    model_dir: Path = DEFAULT_MODEL_DIR
    target_lufs: float = -14.0
    ceiling_dbtp: float = -1.0
    # None removes the audience entirely; a value such as -15 keeps it that
    # many LU under the lead vocal for a "live album" feel.
    crowd_db: float | None = None
    reference: Path | None = None
    balance_strength: float = 0.6
    stem_gains_db: dict[str, float] = field(default_factory=dict)
    tonal_strength: float = 0.5
    declip: bool = True
    output_format: str = "wav"
    bit_depth: int = 24
    export_stems: bool = True
    export_raw_stems: bool = False
    chunk_seconds: float | None = None
    # Overrides the preset's Roformer overlap: higher is cleaner and slower.
    roformer_overlap: int | None = None
    keep_work_files: bool = False
    verbose_models: bool = False


@dataclass
class RemasterResult:
    master_path: Path
    stem_paths: dict[str, Path]
    report_path: Path
    report: dict


def remaster(
    input_path: str | Path,
    output_dir: str | Path,
    settings: RemasterSettings | None = None,
    backend: SeparationBackend | None = None,
    progress: Callable[[str], None] | None = None,
) -> RemasterResult:
    settings = settings or RemasterSettings()
    progress = progress or log.info
    input_path = Path(input_path)
    title = input_path.stem
    out_dir = Path(output_dir) / title
    sr = SAMPLE_RATE
    started = time.perf_counter()
    timings: dict[str, float] = {}

    plan = build_plan(settings.preset, settings.stems, **settings.model_overrides)
    reference = load_audio(settings.reference) if settings.reference else None

    progress(f"Loading {input_path.name}")
    audio = load_audio(input_path)
    repaired = 0
    if settings.declip:
        audio, repaired = declip(audio)
        if repaired:
            progress(f"Rebuilt {repaired} clipped samples")
    audio = remove_rumble(audio, sr)

    t = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="concert-remaster-") as tmp:
        work_dir = out_dir / "work" if settings.keep_work_files else Path(tmp)
        if backend is None:
            preset = PRESETS[settings.preset]
            backend = AudioSeparatorBackend(
                settings.model_dir,
                work_dir,
                chunk_seconds=settings.chunk_seconds,
                roformer_overlap=settings.roformer_overlap or preset.roformer_overlap,
                demucs_shifts=preset.demucs_shifts,
                log_level=logging.INFO if settings.verbose_models else logging.WARNING,
                keep_files=settings.keep_work_files,
            )
        separation = separate_concert(audio, sr, plan, backend, progress)
    timings["separation"] = time.perf_counter() - t

    t = time.perf_counter()
    progress("Applying studio processing to each stem")
    raw_stems = dict(separation.stems)
    crowd = raw_stems.pop("crowd", None)
    processed = {name: process_stem(stem, sr, profile_for(name)) for name, stem in raw_stems.items()}

    balance = auto_balance(processed, sr, settings.balance_strength, user_gains_db=settings.stem_gains_db)
    gains = dict(balance.gains_db)
    loudness = dict(balance.loudness_lufs)
    if crowd is not None and settings.crowd_db is not None and balance.reference is not None:
        processed["crowd"] = process_stem(crowd, sr, profile_for("crowd"))
        reference_level = loudness[balance.reference] + gains[balance.reference]
        loudness["crowd"] = integrated_lufs(processed["crowd"], sr)
        if math.isfinite(loudness["crowd"]):
            gains["crowd"] = reference_level + settings.crowd_db - loudness["crowd"]
        else:
            processed.pop("crowd")
    mix = sum_stems(processed, gains)
    timings["mixing"] = time.perf_counter() - t

    t = time.perf_counter()
    progress("Mastering")
    mastered, master_report = master(
        mix,
        sr,
        target_lufs=settings.target_lufs,
        ceiling_dbtp=settings.ceiling_dbtp,
        reference=reference,
        tonal_strength=settings.tonal_strength,
    )
    timings["mastering"] = time.perf_counter() - t

    ext = settings.output_format.lower().lstrip(".")
    master_path = save_audio(out_dir / f"{title} (Remastered).{ext}", mastered, sr, settings.bit_depth)

    stem_paths: dict[str, Path] = {}
    if settings.export_stems:
        balanced = {name: stem * np.float32(10 ** (gains.get(name, 0.0) / 20)) for name, stem in processed.items()}
        for name, stem in _common_headroom(balanced, mix_peak=float(np.abs(mix).max())).items():
            stem_paths[name] = save_audio(out_dir / "stems" / f"{name}.{ext}", stem, sr, settings.bit_depth)
    if settings.export_raw_stems:
        raw = dict(separation.stems, **separation.removed)
        for name, stem in _common_headroom(raw, mix_peak=float(np.abs(audio).max())).items():
            stem_paths[f"raw/{name}"] = save_audio(out_dir / "stems_raw" / f"{name}.{ext}", stem, sr, settings.bit_depth)

    report = {
        "input": str(input_path),
        "duration_seconds": round(audio.shape[-1] / sr, 2),
        "preset": settings.preset,
        "models": asdict(plan),
        "declipped_samples": repaired,
        "stems": {
            name: {
                "loudness_lufs": _finite(loudness.get(name)),
                "gain_db": round(gains.get(name, 0.0), 2),
                "ghost": name in balance.ghosts,
            }
            for name in processed
        },
        "balance_reference": balance.reference,
        "master": {key: _finite(value) for key, value in asdict(master_report).items()},
        "target_lufs": settings.target_lufs,
        "ceiling_dbtp": settings.ceiling_dbtp,
        "timings_seconds": {key: round(value, 1) for key, value in timings.items()},
        "total_seconds": round(time.perf_counter() - started, 1),
    }
    report_path = out_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2))
    progress(f"Done: {master_path}")
    return RemasterResult(master_path, stem_paths, report_path, report)


def _common_headroom(stems: dict[str, np.ndarray], mix_peak: float, peak: float = 0.89) -> dict[str, np.ndarray]:
    """Scale every stem by one factor so their sum peaks near -1 dBFS and none clips.

    One shared factor keeps the stems balanced exactly as in the mix, so they
    line up when dropped into a DAW together.
    """
    loudest = max((float(np.abs(stem).max()) for stem in stems.values()), default=0.0)
    if loudest == 0.0:
        return stems
    scale = min(peak / max(mix_peak, 1e-9), peak / loudest)
    return {name: stem * np.float32(scale) for name, stem in stems.items()}


def _finite(value):
    if isinstance(value, float):
        return round(value, 2) if math.isfinite(value) else None
    if isinstance(value, list):
        return [_finite(v) for v in value]
    return value
