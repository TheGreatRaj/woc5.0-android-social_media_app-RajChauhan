"""The artist talking: making the voice clear, and writing down what was said."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from pedalboard import Compressor, HighShelfFilter, PeakFilter, Pedalboard

from .analysis import active_level_db
from .paths import models_dir
from .restoration import highpass, spectral_denoise
from .settings import SpeechSettings
from .stem_processing import deess

log = logging.getLogger(__name__)


def enhance_voice(voice: np.ndarray, sample_rate: int, settings: SpeechSettings) -> np.ndarray:
    """Dialogue chain for stage banter: rumble out, noise down, warm and present, even level."""
    x = highpass(np.ascontiguousarray(voice, dtype=np.float32), sample_rate, 80.0, order=2)
    if settings.denoise > 0:
        x = spectral_denoise(x, sample_rate, settings.denoise, max_reduction_db=18.0)
    loud = active_level_db(x, sample_rate)
    if loud is None:
        return x
    board = Pedalboard([
        PeakFilter(cutoff_frequency_hz=200.0, gain_db=settings.warmth_db, q=0.8),
        PeakFilter(cutoff_frequency_hz=450.0, gain_db=-1.5, q=1.0),  # boxiness of a hand-held mic in a hall
        PeakFilter(cutoff_frequency_hz=3000.0, gain_db=settings.presence_db, q=0.9),
        HighShelfFilter(cutoff_frequency_hz=9000.0, gain_db=1.0),
        Compressor(threshold_db=loud - 14.0, ratio=settings.compression_ratio, attack_ms=8.0, release_ms=120.0),
    ])
    x = board(x, sample_rate)
    x = deess(x, sample_rate)
    # A mic voice is centred.
    return np.repeat(x.mean(axis=0, keepdims=True), 2, axis=0).astype(np.float32)


class Transcriber:
    """Offline speech-to-text with faster-whisper; loads the model on first use.

    Uses the NVIDIA GPU when available (float16), otherwise the CPU (int8).
    Models are stored in ``models/whisper``.
    """

    def __init__(self, model: str = "large-v3", language: str = "auto", device: str = "auto", model_dir: str | Path | None = None):
        self.model_name = model
        self.language = None if language in ("", "auto") else language
        self.device = device
        self.model_dir = Path(model_dir) if model_dir else models_dir() / "whisper"
        self._model = None

    def _load(self):
        if self._model is None:
            from faster_whisper import WhisperModel

            device = "cpu"
            if self.device in ("auto", "cuda"):
                try:
                    import ctranslate2

                    if ctranslate2.get_cuda_device_count() > 0:
                        device = "cuda"
                except Exception:
                    device = "cpu"
            compute = "float16" if device == "cuda" else "int8"
            self.model_dir.mkdir(parents=True, exist_ok=True)
            try:
                self._model = WhisperModel(self.model_name, device=device, compute_type=compute, download_root=str(self.model_dir))
            except Exception as exc:
                if device != "cuda":
                    raise
                log.warning("Whisper on the GPU failed (%s); using the CPU", exc)
                self._model = WhisperModel(self.model_name, device="cpu", compute_type="int8", download_root=str(self.model_dir))
        return self._model

    def segments(self, audio: np.ndarray, sample_rate: int, offset: float = 0.0) -> list[dict]:
        """Timed phrases: ``[{"start", "end", "text"}]`` with times offset by ``offset`` seconds."""
        import librosa

        mono = audio.mean(axis=0) if audio.ndim == 2 else audio
        y = librosa.resample(np.asarray(mono, dtype=np.float32), orig_sr=sample_rate, target_sr=16000)
        peak = float(np.abs(y).max()) if y.size else 0.0
        if peak == 0.0:
            return []
        parts, info = self._load().transcribe(y / peak * 0.8, language=self.language, vad_filter=True, beam_size=5,
                                              condition_on_previous_text=False)
        return [{"start": round(offset + p.start, 2), "end": round(offset + p.end, 2), "text": p.text.strip()}
                for p in parts if p.text.strip() and p.no_speech_prob < 0.6]

    def transcribe(self, audio: np.ndarray, sample_rate: int = 44100, max_seconds: float | None = None) -> str:
        if max_seconds:
            audio = audio[..., : int(max_seconds * sample_rate)]
        return " ".join(s["text"] for s in self.segments(audio, sample_rate))


def srt(entries: list[dict]) -> str:
    """SubRip subtitles from ``[{"start", "end", "text"}]``."""

    def stamp(seconds: float) -> str:
        ms = int(round(seconds * 1000))
        h, ms = divmod(ms, 3_600_000)
        m, ms = divmod(ms, 60_000)
        s, ms = divmod(ms, 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    blocks = [f"{i}\n{stamp(e['start'])} --> {stamp(e['end'])}\n{e['text']}\n" for i, e in enumerate(entries, 1)]
    return "\n".join(blocks)
