"""Reading and writing audio.

Inside the pipeline audio is always float32 with shape ``(channels, samples)``,
stereo, at :data:`SAMPLE_RATE`. That is the layout pedalboard uses and the rate
every separation model was trained on.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

SAMPLE_RATE = 44100

# Container formats phones usually record concerts in. pedalboard cannot open
# these on Linux, so they go straight to ffmpeg.
_VIDEO_SUFFIXES = {".mp4", ".mov", ".m4a", ".m4v", ".3gp", ".mkv", ".webm", ".aac"}


class AudioLoadError(RuntimeError):
    """Raised when an input file cannot be decoded."""


def load_audio(path: str | Path, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Decode any audio or video file to stereo float32 at ``sample_rate``."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"No such file: {path}")

    audio = None
    error: Exception | None = None
    if path.suffix.lower() not in _VIDEO_SUFFIXES:
        try:
            audio = _load_with_pedalboard(path, sample_rate)
        except Exception as exc:  # unsupported codec, corrupt header, ...
            error = exc
    if audio is None:
        if shutil.which("ffmpeg") is None:
            reason = f": {error}" if error else ""
            raise AudioLoadError(
                f"Could not decode {path.name}{reason}. Install ffmpeg to open video and compressed formats."
            )
        audio = _load_with_ffmpeg(path, sample_rate)

    if audio.size == 0:
        raise AudioLoadError(f"{path.name} contains no audio")
    return to_stereo(audio)


def _load_with_pedalboard(path: Path, sample_rate: int) -> np.ndarray:
    from pedalboard.io import AudioFile

    with AudioFile(str(path)).resampled_to(sample_rate) as f:
        return np.asarray(f.read(f.frames), dtype=np.float32)


def _load_with_ffmpeg(path: Path, sample_rate: int) -> np.ndarray:
    cmd = [
        "ffmpeg", "-nostdin", "-v", "error", "-i", str(path),
        "-vn", "-ac", "2", "-ar", str(sample_rate), "-f", "f32le", "pipe:1",
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        raise AudioLoadError(f"ffmpeg could not decode {path.name}: {result.stderr.decode(errors='replace').strip()}")
    return np.frombuffer(result.stdout, dtype=np.float32).reshape(-1, 2).T.copy()


def to_stereo(audio: np.ndarray) -> np.ndarray:
    """Return a contiguous float32 ``(2, samples)`` array."""
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 1:
        audio = audio[None, :]
    if audio.shape[0] == 1:
        audio = np.repeat(audio, 2, axis=0)
    elif audio.shape[0] > 2:
        audio = audio[:2]
    return np.ascontiguousarray(audio)


def save_audio(path: str | Path, audio: np.ndarray, sample_rate: int = SAMPLE_RATE, bit_depth: int | str = 24) -> Path:
    """Write ``audio`` to ``path``; the format follows the file extension.

    ``bit_depth`` is 16, 24 or ``"float"`` for WAV/FLAC. 16-bit output gets TPDF
    dither so quiet fades don't turn into quantisation grit. MP3 is written at
    320 kbps.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    audio = np.asarray(audio, dtype=np.float32)
    suffix = path.suffix.lower()

    if suffix == ".mp3":
        from pedalboard.io import AudioFile

        with AudioFile(str(path), "w", sample_rate, audio.shape[0], quality="320") as f:
            f.write(np.clip(audio, -1.0, 1.0))
        return path

    if suffix not in {".wav", ".flac", ".aiff", ".aif"}:
        raise ValueError(f"Unsupported output format: {suffix} (use .wav, .flac, .aiff or .mp3)")

    if bit_depth == "float":
        if suffix != ".wav":
            raise ValueError("Float output is only supported for WAV files")
        sf.write(str(path), audio.T, sample_rate, subtype="FLOAT")
        return path

    if bit_depth not in (16, 24):
        raise ValueError(f"bit_depth must be 16, 24 or 'float', got {bit_depth!r}")
    if bit_depth == 16:
        rng = np.random.default_rng(0)
        lsb = 1.0 / 32768.0
        audio = audio + (rng.random(audio.shape, dtype=np.float32) - rng.random(audio.shape, dtype=np.float32)) * lsb
    sf.write(str(path), np.clip(audio, -1.0, 1.0).T, sample_rate, subtype=f"PCM_{bit_depth}")
    return path
