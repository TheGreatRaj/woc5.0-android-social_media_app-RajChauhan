"""On-device AI source separation.

The models are the community state of the art from the Ultimate Vocal Remover
project (BS/Mel-Band Roformer, MDX-Net, VR) and Meta's Demucs, all run locally
through the ``audio-separator`` package on NVIDIA (CUDA), AMD (DirectML) or
the CPU. Weights are downloaded once to the model folder; after that no
network access is needed.

A concert goes through a graph of passes; each reads one stem and writes new
ones::

    source ─ crowd ──► music ─ vocals ──► instrumental ─ instruments ──► drums, bass, guitar, piano, other_all
                 └► crowd          └► vocals_wet ─ dereverb ─ denoise ─► vocals ─ lead/backing
    drums ─ drum kit ──► kick, snare, toms, hihat, ride, crash
    other_all ─ woodwinds ──► woodwinds, other_rest
"""

from __future__ import annotations

import logging
import os
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol

import numpy as np
import soundfile as sf

from .audio_io import SAMPLE_RATE, ensure_ffmpeg, to_stereo
from .settings import ModelSettings

log = logging.getLogger(__name__)


class SeparationError(RuntimeError):
    pass


class SeparationBackend(Protocol):
    def separate(self, audio: np.ndarray, sample_rate: int, model: str) -> dict[str, np.ndarray]:
        """Split ``audio`` with ``model``; keys are normalised stem names."""


def normalize_stem_name(name: str) -> str:
    """``"No Crowd"`` -> ``"nocrowd"``, ``"Vocals"`` -> ``"vocals"``."""
    return re.sub(r"[^a-z]", "", name.lower())


def split_models(spec: str) -> list[str]:
    """``"a.ckpt + b.ckpt"`` -> ``["a.ckpt", "b.ckpt"]`` (an ensemble)."""
    return [part.strip() for part in spec.split("+") if part.strip()]


class AudioSeparatorBackend:
    """Runs models through ``audio-separator``, exchanging float WAVs in ``work_dir``.

    audio-separator rescales any output stem whose peak exceeds 1.0, which
    would wreck the level relationships the remix depends on. Inputs are
    therefore written as 32-bit float with peaks at :attr:`HEADROOM` and every
    stem is scaled back afterwards, which keeps levels exact.

    On DirectML (AMD GPUs on Windows) some model operations are unsupported;
    a model that fails there is retried on the CPU and stays on the CPU.
    """

    HEADROOM = 0.35

    def __init__(
        self,
        model_dir: str | Path,
        work_dir: str | Path,
        device: str = "auto",
        half_precision: bool = False,
        roformer_overlap: int | None = 4,
        demucs_shifts: int = 2,
        ensemble_algorithm: str = "avg_fft",
        log_level: int = logging.ERROR,
        keep_files: bool = False,
    ):
        self.model_dir = Path(model_dir).expanduser().resolve()
        self.work_dir = Path(work_dir).resolve()
        self.device = device
        self.half_precision = half_precision
        self.roformer_overlap = roformer_overlap
        self.demucs_shifts = demucs_shifts
        self.ensemble_algorithm = ensemble_algorithm
        self.log_level = log_level
        self.keep_files = keep_files
        self._separators: dict[bool, object] = {}  # keyed by "accelerated"
        self._loaded: dict[bool, str | None] = {True: None, False: None}
        self._cpu_models: set[str] = set()
        self._verified: set[str] = set()  # model files checked complete in this run
        self._calls = 0
        if device == "cpu":
            # Must happen before torch initialises CUDA.
            os.environ["CUDA_VISIBLE_DEVICES"] = ""

    def device_label(self) -> str:
        """Where the models will run, for progress messages (without loading a model)."""
        if self.device == "cpu":
            return "CPU"
        try:
            import torch

            if self.device in ("auto", "cuda") and torch.cuda.is_available():
                return f"GPU: {torch.cuda.get_device_name(0)}"
        except Exception:
            pass
        if self.device == "directml":
            return "GPU: DirectML"
        return "CPU (no usable GPU)"

    def device_description(self) -> str:
        separator = self._get_separator(accelerated=True)
        return str(getattr(separator, "torch_device", "cpu"))

    def _get_separator(self, accelerated: bool):
        if accelerated not in self._separators:
            try:
                from audio_separator.separator import Separator
            except ImportError as exc:
                raise SeparationError(
                    "The AI separation models need audio-separator. Run setup.bat, or "
                    "`pip install \"audio-separator[gpu]\"` (NVIDIA) / `[dml]` (AMD) / `[cpu]`."
                ) from exc
            ensure_ffmpeg()
            self.model_dir.mkdir(parents=True, exist_ok=True)
            self.work_dir.mkdir(parents=True, exist_ok=True)
            # The non-accelerated separator only exists as the DirectML fallback, which
            # runs on machines without CUDA, so leaving DirectML off gives the CPU.
            use_directml = accelerated and self.device == "directml"
            import torch

            on_cuda = accelerated and self.device in ("auto", "cuda") and torch.cuda.is_available()
            self._separators[accelerated] = Separator(
                log_level=self.log_level,
                model_file_dir=str(self.model_dir),
                output_dir=str(self.work_dir),
                output_format="WAV",
                normalization_threshold=1.0,
                sample_rate=SAMPLE_RATE,
                use_soundfile=True,
                use_directml=use_directml,
                use_autocast=bool(on_cuda and self.half_precision),
                ensemble_algorithm=self.ensemble_algorithm,
                mdxc_params={
                    "segment_size": 256,
                    "override_model_segment_size": False,
                    "batch_size": None,
                    "overlap": self.roformer_overlap,
                    "pitch_shift": 0,
                },
                demucs_params={"segment_size": "Default", "shifts": self.demucs_shifts, "overlap": 0.25, "segments_enabled": True},
            )
        return self._separators[accelerated]

    def separate(self, audio: np.ndarray, sample_rate: int, model: str) -> dict[str, np.ndarray]:
        accelerated = self.device != "cpu" and model not in self._cpu_models
        try:
            return self._separate_with(accelerated, audio, sample_rate, model)
        except SeparationError:
            raise
        except Exception as exc:
            if not accelerated or self.device != "directml":
                raise
            log.warning("%s failed on DirectML (%s); retrying on the CPU", model, exc)
            self._cpu_models.add(model)
            return self._separate_with(False, audio, sample_rate, model)

    def _drop_damaged(self, models: list[str]) -> list[str]:
        """Delete damaged model files so they are downloaded again; returns their names."""
        dropped = []
        for name in models:
            path = self.model_dir / name
            if name in self._verified or not path.exists():
                continue
            if model_file_ok(path):
                self._verified.add(name)
            else:
                log.warning("Model file %s is damaged or incomplete; downloading it again", name)
                path.unlink(missing_ok=True)
                dropped.append(name)
        return dropped

    def _load(self, separator, models: list[str]) -> None:
        self._drop_damaged(models)
        target = models if len(models) > 1 else models[0]
        try:
            separator.load_model(model_filename=target)
        except Exception as exc:
            if not any(word in str(exc).lower() for word in _DAMAGED):
                raise
            # Damaged in a way the quick check missed: fetch every file of this model again, once.
            for name in models:
                (self.model_dir / name).unlink(missing_ok=True)
                self._verified.discard(name)
            try:
                separator.load_model(model_filename=target)
            except Exception as again:
                raise SeparationError(
                    f"The AI model {', '.join(models)} is damaged and could not be downloaded again ({again}). "
                    "Connect to the internet and press Continue, or run setup.bat again."
                ) from again

    def _separate_with(self, accelerated: bool, audio: np.ndarray, sample_rate: int, model: str) -> dict[str, np.ndarray]:
        separator = self._get_separator(accelerated)
        models = split_models(model)
        if not models:
            raise SeparationError("No model given")
        if self._loaded[accelerated] != model:
            self._load(separator, models)
            self._loaded[accelerated] = model

        peak = float(np.abs(audio).max())
        scale = self.HEADROOM / peak if peak > 0 else 1.0
        self._calls += 1
        tag = f"pass{self._calls:04d}"
        in_path = self.work_dir / f"{tag}.wav"
        sf.write(str(in_path), (audio * scale).T, sample_rate, subtype="FLOAT")

        stems: dict[str, np.ndarray] = {}
        try:
            for output in separator.separate(str(in_path)):
                path = Path(output)
                if not path.is_absolute() and not path.exists():
                    path = self.work_dir / path.name
                match = re.match(rf"{tag}_\((.+?)\)", path.name)
                name = normalize_stem_name(match.group(1) if match else path.stem)
                data, sr = sf.read(str(path), dtype="float32", always_2d=True)
                if sr != sample_rate:
                    raise SeparationError(f"{model} returned {sr} Hz audio, expected {sample_rate} Hz")
                if np.abs(data).max() >= 0.999:
                    log.warning("%s stem %r hit full scale; its level may be off", model, name)
                stem = _fit_length(to_stereo(data.T) / scale, audio.shape[-1])
                stems[name] = stems[name] + stem if name in stems else stem
                if not self.keep_files:
                    path.unlink(missing_ok=True)
        finally:
            if not self.keep_files:
                in_path.unlink(missing_ok=True)
        if not stems:
            raise SeparationError(f"{model} produced no output")
        return stems


_WEIGHTS = (".ckpt", ".pth", ".pt", ".th", ".bin", ".onnx")
_DAMAGED = ("central directory", "corrupt", "pytorchstreamreader", "unpickl", "unexpected eof", "ran out of input",
            "invalid load key", "protobuf", "incomplete")


def model_file_ok(path: Path) -> bool:
    """Is a downloaded model file complete?

    The model downloader writes straight to the final file name, so an interrupted
    download (setup closed, connection dropped) leaves a truncated file that looks
    present and is never fetched again. PyTorch checkpoints are zip archives whose
    directory sits at the very end, so a cut-off file is missing it; older pickle
    checkpoints and ONNX files are test-loaded.
    """
    path = Path(path)
    try:
        suffix = path.suffix.lower()
        if suffix not in _WEIGHTS:
            return path.stat().st_size > 0  # configs are small and use custom YAML tags; only weights are checked
        if path.stat().st_size < 1024:
            return False
        if suffix in (".ckpt", ".pth", ".pt", ".th", ".bin"):
            if zipfile.is_zipfile(path):
                with zipfile.ZipFile(path) as archive:
                    last = max(archive.infolist(), key=lambda i: i.header_offset, default=None)
                    return last is None or last.header_offset + last.compress_size <= path.stat().st_size
            import torch

            torch.load(str(path), map_location="cpu", weights_only=False)
            return True
        import onnx

        onnx.load(str(path), load_external_data=False)
        return True
    except Exception:
        return False


def _fit_length(audio: np.ndarray, length: int) -> np.ndarray:
    if audio.shape[-1] >= length:
        return np.ascontiguousarray(audio[:, :length], dtype=np.float32)
    return np.pad(audio, ((0, 0), (0, length - audio.shape[-1]))).astype(np.float32)


# --- the pass graph ----------------------------------------------------------

_CROWD = {"crowd"}
_VOCALS = {"vocals", "vocal"}
_DRY = {"noreverb", "dry", "noecho"}
_CLEAN = {"dry", "nonoise", "clean"}
_INSTRUMENTS = {"drums": "drums", "bass": "bass", "guitar": "guitar", "piano": "piano", "other": "other_all"}
_KIT = {"kick": "kick", "snare": "snare", "toms": "toms", "hh": "hihat", "hihat": "hihat", "ride": "ride", "crash": "crash", "cymbals": "cymbals"}


def pick(outputs: dict[str, np.ndarray], wanted: set[str], source: np.ndarray, what: str, model: str):
    """Return (the wanted stem, everything else) from a separation pass."""
    key = next((name for name in outputs if name in wanted), None)
    if key is None:
        raise SeparationError(f"{model} gave no {what} stem (got: {', '.join(outputs)})")
    rest = [audio for name, audio in outputs.items() if name != key]
    remainder = np.sum(rest, axis=0) if rest else source - outputs[key]
    return outputs[key], remainder.astype(np.float32)


@dataclass(frozen=True)
class PassSpec:
    name: str
    input: str
    model: str
    outputs: tuple[str, ...]
    split: Callable[[dict[str, np.ndarray], np.ndarray, str], dict[str, np.ndarray]]
    label: str


def _split_two(target_set, target_name, rest_name, what):
    def split(outputs, source, model):
        target, rest = pick(outputs, target_set, source, what, model)
        return {target_name: target, rest_name: rest}

    return split


def _split_instruments(vocals_already_isolated: bool):
    def split(outputs, source, model):
        result: dict[str, np.ndarray] = {}
        for name, audio in outputs.items():
            if name in _VOCALS:
                # Vocal left in the instrumental stays in the mix, but out of the clean vocal stem.
                target = "other_all" if vocals_already_isolated else "vocals_wet"
            else:
                target = _INSTRUMENTS.get(name, "other_all")
            result[target] = result[target] + audio if target in result else audio
        for name in ("drums", "bass", "guitar", "piano", "other_all"):
            result.setdefault(name, np.zeros_like(source))
        if not vocals_already_isolated:
            result.setdefault("vocals_wet", np.zeros_like(source))
        return result

    return split


def _split_karaoke(outputs, source, model):
    lead, backing = pick(outputs, _VOCALS, source, "lead vocal", model)
    return {"lead_vocals": lead, "backing_vocals": backing}


def _split_kit(outputs, source, model):
    result: dict[str, np.ndarray] = {}
    for name, audio in outputs.items():
        target = _KIT.get(name, name)
        result[target] = result[target] + audio if target in result else audio
    return result


def build_passes(models: ModelSettings) -> list[PassSpec]:
    """Turn the model settings into the ordered list of separation passes."""
    passes: list[PassSpec] = []
    music = "source"
    if models.crowd_enabled and models.crowd_model:
        passes.append(PassSpec("crowd", "source", models.crowd_model, ("crowd", "music"),
                               _split_two(_CROWD, "crowd", "music", "crowd"), "Removing crowd noise"))
        music = "music"

    vocal_ensemble = bool(models.vocal_model.strip())
    band = music
    if vocal_ensemble:
        passes.append(PassSpec("vocals", music, models.vocal_model, ("vocals_wet", "instrumental"),
                               _split_two(_VOCALS, "vocals_wet", "instrumental", "vocal"), "Isolating vocals"))
        band = "instrumental"
    if not models.instrument_model:
        raise ValueError("An instrument model is required")
    outputs = ("drums", "bass", "guitar", "piano", "other_all") + (() if vocal_ensemble else ("vocals_wet",))
    passes.append(PassSpec("instruments", band, models.instrument_model, outputs,
                           _split_instruments(vocal_ensemble), "Splitting instruments"))

    vocals = "vocals_wet"
    if models.dereverb_enabled and models.dereverb_model:
        passes.append(PassSpec("dereverb", vocals, models.dereverb_model, ("vocals_dry", "vocal_reverb"),
                               _split_two(_DRY, "vocals_dry", "vocal_reverb", "dry"), "Removing venue echo from vocals"))
        vocals = "vocals_dry"
    if models.denoise_enabled and models.denoise_model:
        passes.append(PassSpec("denoise", vocals, models.denoise_model, ("vocals_clean", "vocal_noise"),
                               _split_two(_CLEAN, "vocals_clean", "vocal_noise", "clean"), "Denoising vocals"))
        vocals = "vocals_clean"
    if models.lead_backing_enabled and models.lead_backing_model:
        passes.append(PassSpec("lead_backing", vocals, models.lead_backing_model, ("lead_vocals", "backing_vocals"),
                               _split_karaoke, "Splitting lead and backing vocals"))
    if models.drum_kit_enabled and models.drum_kit_model:
        passes.append(PassSpec("drum_kit", "drums", models.drum_kit_model, ("kick", "snare", "toms", "hihat", "ride", "crash"),
                               _split_kit, "Splitting the drum kit"))
    if models.woodwinds_enabled and models.woodwind_model:
        passes.append(PassSpec("woodwinds", "other_all", models.woodwind_model, ("woodwinds", "other_rest"),
                               _split_two({"woodwinds"}, "woodwinds", "other_rest", "woodwinds"), "Extracting flute and woodwinds"))
    return passes


def stem_aliases(passes: list[PassSpec]) -> dict[str, str]:
    """Map friendly stem names to the pass outputs that hold them."""
    produced = {out for p in passes for out in p.outputs}
    aliases = {}
    for alias, candidates in {
        "vocals": ("vocals_clean", "vocals_dry", "vocals_wet"),
        "other": ("other_rest", "other_all"),
    }.items():
        aliases[alias] = next(c for c in candidates if c in produced)
    for name in ("crowd", "drums", "bass", "guitar", "piano", "woodwinds", "lead_vocals", "backing_vocals",
                 "kick", "snare", "toms", "hihat", "ride", "crash", "vocal_reverb", "vocal_noise"):
        if name in produced:
            aliases[name] = name
    return aliases


# Stems summed into the mix (the drum kit pieces and removed reverb/noise are export-only).
MIX_STEMS = ("vocals", "lead_vocals", "backing_vocals", "drums", "bass", "guitar", "piano", "woodwinds", "other")


KIT_STEMS = ("kick", "snare", "toms", "hihat", "ride", "crash")


def mix_stems(aliases: dict[str, str], drum_kit: bool = False) -> list[str]:
    names = [n for n in MIX_STEMS if n in aliases]
    if "lead_vocals" in names:
        names.remove("vocals")  # lead + backing replace the combined vocal
    if drum_kit and "drums" in names and all(k in aliases for k in KIT_STEMS):
        i = names.index("drums")
        names[i:i + 1] = list(KIT_STEMS)  # every drum on its own mixer track
    return names


EXPORT_STEMS = MIX_STEMS + KIT_STEMS + ("crowd",)


def all_models(models: ModelSettings) -> list[str]:
    """Every model file the settings need, for pre-downloading."""
    names: list[str] = []
    for p in build_passes(models):
        names.extend(split_models(p.model))
    return list(dict.fromkeys(names))
