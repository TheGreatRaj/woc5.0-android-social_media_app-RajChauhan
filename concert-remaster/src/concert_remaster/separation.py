"""On-device AI source separation.

The models are the community state of the art from the Ultimate Vocal Remover
project (BS/Mel-Band Roformer, MDX-Net, VR) and Meta's Demucs, all run locally
through the ``audio-separator`` package on CPU, CUDA or Apple Silicon (MPS).
Weights are downloaded once to the model directory; after that no network
access is needed.

A concert recording goes through up to five passes::

    recording -> crowd removal -> vocal isolation -> instrument split
                                       |
                                       +-> de-reverb -> denoise   (vocal only)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Protocol

import numpy as np
import soundfile as sf

from .audio_io import SAMPLE_RATE, to_stereo

log = logging.getLogger(__name__)

ROFORMER_CROWD = "mel_band_roformer_crowd_aufr33_viperx_sdr_8.7144.ckpt"
MDX_CROWD = "UVR-MDX-NET_Crowd_HQ_1.onnx"
BS_ROFORMER_VOCALS = "model_bs_roformer_ep_317_sdr_12.9755.ckpt"
MEL_ROFORMER_VOCALS = "vocals_mel_band_roformer.ckpt"
MDX_VOCALS = "UVR-MDX-NET-Inst_HQ_4.onnx"
ROFORMER_DEREVERB = "dereverb_mel_band_roformer_anvuew_sdr_19.1729.ckpt"
VR_DEREVERB = "UVR-DeEcho-DeReverb.pth"
ROFORMER_DENOISE = "denoise_mel_band_roformer_aufr33_sdr_27.9959.ckpt"


@dataclass(frozen=True)
class Preset:
    description: str
    crowd: str | None
    vocals: str | None
    dereverb: str | None
    denoise: str | None
    four_stem: str
    six_stem: str
    # Speed knobs. Roformers slide an ~8 s window with this many overlapping
    # passes (None keeps each model's own default, usually 4); Demucs averages
    # this many time-shifted passes.
    roformer_overlap: int | None = 2
    demucs_shifts: int = 1


PRESETS: dict[str, Preset] = {
    "best": Preset(
        description="Roformer models for every stage. Cleanest result; wants a GPU for full songs.",
        crowd=ROFORMER_CROWD,
        vocals=BS_ROFORMER_VOCALS,
        dereverb=ROFORMER_DEREVERB,
        denoise=ROFORMER_DENOISE,
        four_stem="htdemucs_ft.yaml",
        six_stem="htdemucs_6s.yaml",
        roformer_overlap=None,
        demucs_shifts=2,
    ),
    "balanced": Preset(
        description="Roformer crowd removal and vocals, lighter de-reverb, 2 overlap passes. Practical on CPU.",
        crowd=ROFORMER_CROWD,
        vocals=MEL_ROFORMER_VOCALS,
        dereverb=VR_DEREVERB,
        denoise=None,
        four_stem="htdemucs_ft.yaml",
        six_stem="htdemucs_6s.yaml",
    ),
    "fast": Preset(
        description="MDX-Net crowd removal (aggressive, can take some music with it) and one Demucs pass. Rough previews.",
        crowd=MDX_CROWD,
        vocals=None,
        dereverb=None,
        denoise=None,
        four_stem="htdemucs.yaml",
        six_stem="htdemucs_6s.yaml",
    ),
}


@dataclass(frozen=True)
class ModelPlan:
    """Which model runs at each stage; ``None`` skips the stage."""

    crowd: str | None
    vocals: str | None
    instruments: str | None
    dereverb: str | None
    denoise: str | None


def build_plan(preset: str = "balanced", stems: int = 6, **overrides: str | None) -> ModelPlan:
    """Resolve a preset and stem count into concrete models.

    ``overrides`` replace individual stages; pass the string ``"none"`` to
    switch a stage off.
    """
    if preset not in PRESETS:
        raise ValueError(f"Unknown preset {preset!r}; choose from {', '.join(PRESETS)}")
    if stems not in (2, 4, 6):
        raise ValueError("stems must be 2, 4 or 6")
    spec = PRESETS[preset]
    instruments = {2: None, 4: spec.four_stem, 6: spec.six_stem}[stems]
    vocals = spec.vocals if spec.vocals or instruments else MDX_VOCALS
    plan = ModelPlan(spec.crowd, vocals, instruments, spec.dereverb, spec.denoise)

    changes = {}
    for stage, model in overrides.items():
        if model is None:
            continue
        if stage not in ModelPlan.__dataclass_fields__:
            raise ValueError(f"Unknown stage {stage!r}")
        changes[stage] = None if model.lower() == "none" else model
    plan = replace(plan, **changes)
    if plan.vocals is None and plan.instruments is None:
        raise ValueError("At least one of the vocal or instrument models must be enabled")
    return plan


class SeparationError(RuntimeError):
    pass


class SeparationBackend(Protocol):
    def separate(self, audio: np.ndarray, sample_rate: int, model: str) -> dict[str, np.ndarray]:
        """Split ``audio`` with ``model``; keys are normalised stem names."""


def normalize_stem_name(name: str) -> str:
    """``"No Crowd"`` -> ``"nocrowd"``, ``"Vocals"`` -> ``"vocals"``."""
    return re.sub(r"[^a-z]", "", name.lower())


class AudioSeparatorBackend:
    """Runs models through ``audio-separator``, exchanging float WAVs in ``work_dir``.

    audio-separator rescales any output stem whose peak exceeds 1.0, which
    would wreck the level relationships the remix depends on. Inputs are
    therefore written as 32-bit float with peaks at :attr:`HEADROOM` and every
    stem is scaled back afterwards, which keeps levels exact.
    """

    HEADROOM = 0.35

    def __init__(
        self,
        model_dir: str | Path,
        work_dir: str | Path,
        chunk_seconds: float | None = None,
        roformer_overlap: int | None = 2,
        demucs_shifts: int = 1,
        log_level: int = logging.WARNING,
        keep_files: bool = False,
    ):
        self.model_dir = Path(model_dir).expanduser()
        self.work_dir = Path(work_dir)
        self.chunk_seconds = chunk_seconds
        self.roformer_overlap = roformer_overlap
        self.demucs_shifts = demucs_shifts
        self.log_level = log_level
        self.keep_files = keep_files
        self._separator = None
        self._loaded_model: str | None = None
        self._calls = 0

    def _get_separator(self):
        if self._separator is None:
            try:
                from audio_separator.separator import Separator
            except ImportError as exc:
                raise SeparationError(
                    "The AI separation models need audio-separator. Install with "
                    "`pip install \"concert-remaster[cpu]\"` (or `[gpu]` for NVIDIA cards)."
                ) from exc
            self.model_dir.mkdir(parents=True, exist_ok=True)
            self.work_dir.mkdir(parents=True, exist_ok=True)
            self._separator = Separator(
                log_level=self.log_level,
                model_file_dir=str(self.model_dir),
                output_dir=str(self.work_dir),
                output_format="WAV",
                normalization_threshold=1.0,
                sample_rate=SAMPLE_RATE,
                use_soundfile=True,
                chunk_duration=self.chunk_seconds,
                mdxc_params={
                    "segment_size": 256,
                    "override_model_segment_size": False,
                    "batch_size": None,
                    "overlap": self.roformer_overlap,
                    "pitch_shift": 0,
                },
                demucs_params={"segment_size": "Default", "shifts": self.demucs_shifts, "overlap": 0.25, "segments_enabled": True},
            )
        return self._separator

    def separate(self, audio: np.ndarray, sample_rate: int, model: str) -> dict[str, np.ndarray]:
        separator = self._get_separator()
        if self._loaded_model != model:
            separator.load_model(model_filename=model)
            self._loaded_model = model

        peak = float(np.abs(audio).max())
        scale = self.HEADROOM / peak if peak > 0 else 1.0
        self._calls += 1
        tag = f"pass{self._calls:02d}"
        in_path = self.work_dir / f"{tag}.wav"
        sf.write(str(in_path), (audio * scale).T, sample_rate, subtype="FLOAT")

        stems: dict[str, np.ndarray] = {}
        try:
            for output in separator.separate(str(in_path)):
                path = Path(output)
                if not path.is_absolute():
                    path = self.work_dir / path
                match = re.match(rf"{tag}_\((.+?)\)_", path.name)
                name = normalize_stem_name(match.group(1) if match else path.stem)
                data, sr = sf.read(str(path), dtype="float32", always_2d=True)
                if sr != sample_rate:
                    raise SeparationError(f"{model} returned {sr} Hz audio, expected {sample_rate} Hz")
                if np.abs(data).max() >= 0.999:
                    log.warning("%s stem %r hit full scale; its level may be off", model, name)
                stems[name] = _fit_length(to_stereo(data.T) / scale, audio.shape[-1])
                if not self.keep_files:
                    path.unlink(missing_ok=True)
        finally:
            if not self.keep_files:
                in_path.unlink(missing_ok=True)
        if not stems:
            raise SeparationError(f"{model} produced no output")
        return stems


def _fit_length(audio: np.ndarray, length: int) -> np.ndarray:
    if audio.shape[-1] >= length:
        return np.ascontiguousarray(audio[:, :length], dtype=np.float32)
    return np.pad(audio, ((0, 0), (0, length - audio.shape[-1]))).astype(np.float32)


_CROWD = {"crowd"}
_VOCALS = {"vocals", "vocal"}
_DRY = {"noreverb", "dry", "noecho"}
_CLEAN = {"dry", "nonoise", "clean"}
_INSTRUMENTS = {"drums", "bass", "guitar", "piano", "other"}


def _split(outputs: dict[str, np.ndarray], wanted: set[str], source: np.ndarray, what: str, model: str):
    """Return (the wanted stem, everything else) from a separation pass."""
    key = next((name for name in outputs if name in wanted), None)
    if key is None:
        raise SeparationError(f"{model} gave no {what} stem (got: {', '.join(outputs)})")
    rest = [audio for name, audio in outputs.items() if name != key]
    remainder = np.sum(rest, axis=0) if rest else source - outputs[key]
    return outputs[key], remainder.astype(np.float32)


@dataclass
class SeparationResult:
    stems: dict[str, np.ndarray]
    # Material removed along the way (venue reverb, noise); exported with --raw-stems.
    removed: dict[str, np.ndarray]


def separate_concert(
    audio: np.ndarray,
    sample_rate: int,
    plan: ModelPlan,
    backend: SeparationBackend,
    progress: Callable[[str], None] = lambda message: None,
) -> SeparationResult:
    stems: dict[str, np.ndarray] = {}
    removed: dict[str, np.ndarray] = {}
    mix = audio

    if plan.crowd:
        progress("Removing crowd noise")
        crowd, mix = _split(backend.separate(mix, sample_rate, plan.crowd), _CROWD, mix, "crowd", plan.crowd)
        stems["crowd"] = crowd

    vocals = None
    band = mix
    if plan.vocals:
        progress("Isolating vocals")
        vocals, band = _split(backend.separate(mix, sample_rate, plan.vocals), _VOCALS, mix, "vocal", plan.vocals)

    if plan.instruments:
        progress("Splitting instruments")
        for name, stem in backend.separate(band, sample_rate, plan.instruments).items():
            if name in _VOCALS:
                if vocals is None:
                    vocals = stem
                    continue
                # Vocal left over in the instrumental: keep it in the mix but out
                # of the clean vocal stem.
                name = "other"
            elif name not in _INSTRUMENTS:
                log.info("Keeping extra stem %r from %s", name, plan.instruments)
            stems[name] = stems[name] + stem if name in stems else stem
    else:
        stems["instrumental"] = band

    if vocals is not None:
        if plan.dereverb:
            progress("Removing venue reverb from vocals")
            outputs = backend.separate(vocals, sample_rate, plan.dereverb)
            vocals, removed["vocal_reverb"] = _split(outputs, _DRY, vocals, "dry", plan.dereverb)
        if plan.denoise:
            progress("Denoising vocals")
            outputs = backend.separate(vocals, sample_rate, plan.denoise)
            vocals, removed["vocal_noise"] = _split(outputs, _CLEAN, vocals, "clean", plan.denoise)
        stems["vocals"] = vocals

    return SeparationResult(stems, removed)
