"""Reading and writing audio.

Inside the pipeline audio is always float32 with shape ``(channels, samples)``,
stereo, at :data:`SAMPLE_RATE`. That is the layout pedalboard uses and the rate
every separation model was trained on.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Iterator

import numpy as np
import soundfile as sf

from .paths import tools_dir

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
        if ensure_ffmpeg() is None:
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


# --- ffmpeg ------------------------------------------------------------------


def ensure_ffmpeg() -> str | None:
    """Find ffmpeg and make sure it is on PATH (audio-separator calls it by name).

    Looks in the app's tools folder (where setup.bat puts it), then PATH,
    then the binary bundled with the imageio-ffmpeg package.
    """
    exe = "ffmpeg.exe" if os.name == "nt" else "ffmpeg"
    for candidate in sorted(tools_dir().glob(f"ffmpeg*/**/{exe}")) + [tools_dir() / exe]:
        if candidate.is_file():
            _prepend_path(candidate.parent)
            return str(candidate)
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg
    except ImportError:
        return None
    bundled = Path(imageio_ffmpeg.get_ffmpeg_exe())
    # audio-separator runs plain "ffmpeg", so expose the bundled binary under that name.
    shim_dir = tools_dir() / "ffmpeg-bundled"
    shim_dir.mkdir(parents=True, exist_ok=True)
    shim = shim_dir / exe
    if not shim.exists():
        shutil.copy2(bundled, shim)
    _prepend_path(shim_dir)
    return str(shim)


def _prepend_path(folder: Path) -> None:
    folder = str(folder)
    if folder not in os.environ.get("PATH", "").split(os.pathsep):
        os.environ["PATH"] = folder + os.pathsep + os.environ.get("PATH", "")


@lru_cache(maxsize=1)
def _has_soxr() -> bool:
    exe = ensure_ffmpeg()
    if exe is None:
        return False
    out = subprocess.run([exe, "-hide_banner", "-version"], capture_output=True, text=True).stdout
    return "libsoxr" in out


def probe_duration(path: str | Path) -> float | None:
    """Duration in seconds from the container header, or None if unknown."""
    exe = ensure_ffmpeg()
    if exe is None:
        return None
    result = subprocess.run([exe, "-hide_banner", "-nostdin", "-i", str(path)], capture_output=True, text=True, errors="replace")
    match = re.search(r"Duration: (\d+):(\d+):(\d+(?:\.\d+)?)", result.stderr)
    if not match:
        return None
    h, m, s = match.groups()
    return int(h) * 3600 + int(m) * 60 + float(s)


def probe_source(path: str | Path) -> dict:
    """Describe the recording: codec, sample rate, channels and bitrate as reported by ffmpeg."""
    exe = ensure_ffmpeg()
    info = {"duration": probe_duration(path), "sample_rate": None, "channels": None, "codec": None, "bitrate_kbps": None}
    if exe is None:
        return info
    stderr = subprocess.run([exe, "-hide_banner", "-nostdin", "-i", str(path)], capture_output=True, text=True, errors="replace").stderr
    audio = re.search(r"Audio: (\w+)[^,]*, (\d+) Hz, ([^,]+)", stderr)
    if audio:
        info["codec"], info["sample_rate"] = audio.group(1), int(audio.group(2))
        layout = audio.group(3).strip()
        info["channels"] = 1 if layout == "mono" else 2 if layout == "stereo" else layout
    rate = re.search(r"Audio:.*?(\d+) kb/s", stderr)
    if rate:
        info["bitrate_kbps"] = int(rate.group(1))
    # A real picture stream (album art in MP3/M4A files shows up as an "attached pic").
    video = [m for m in re.finditer(r"Video: (\w+)[^\n]*", stderr) if "attached pic" not in m.group(0)]
    info["video"] = video[0].group(1) if video else None
    return info


def remux_video(source: str | Path, soundtrack: str | Path, target: str | Path, bitrate_kbps: int = 320) -> Path:
    """Copy the picture of ``source`` untouched and give it ``soundtrack`` as its sound (AAC)."""
    exe = ensure_ffmpeg()
    if exe is None:
        raise AudioLoadError("ffmpeg is required to write videos.")
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.stem + ".partial" + target.suffix)
    cmd = [exe, "-y", "-nostdin", "-v", "error", "-i", str(source), "-i", str(soundtrack),
           "-map", "0:v:0", "-map", "1:a:0", "-map_metadata", "0", "-c:v", "copy",
           "-c:a", "aac", "-b:a", f"{bitrate_kbps}k", "-ar", "48000"]
    if target.suffix.lower() in (".mp4", ".m4v", ".mov"):
        cmd += ["-movflags", "+faststart"]
    result = subprocess.run(cmd + [str(partial)], capture_output=True, text=True, errors="replace")
    if result.returncode != 0:
        partial.unlink(missing_ok=True)
        raise AudioLoadError(f"Could not write the video: {result.stderr.strip()[-400:]}")
    partial.replace(target)
    return target


def stream_decode(path: str | Path, sample_rate: int = SAMPLE_RATE, block_seconds: float = 30.0) -> Iterator[np.ndarray]:
    """Decode any audio/video file block by block as stereo float32 ``(2, n)``.

    Hours-long recordings never have to fit in memory. Resampling uses the
    SoX resampler when ffmpeg has it (phones record at 48 kHz; the models
    want 44.1 kHz).
    """
    exe = ensure_ffmpeg()
    if exe is None:
        raise AudioLoadError("ffmpeg is required to read recordings. Run setup.bat or install ffmpeg.")
    resample = f"aresample={sample_rate}" + (":resampler=soxr:precision=28" if _has_soxr() else "")
    cmd = [exe, "-nostdin", "-v", "error", "-i", str(path), "-vn", "-map", "0:a:0", "-ac", "2", "-af", resample, "-f", "f32le", "pipe:1"]
    block_bytes = int(block_seconds * sample_rate) * 2 * 4
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        pending = b""
        while True:
            chunk = proc.stdout.read(block_bytes - len(pending))
            if not chunk:
                break
            pending += chunk
            if len(pending) >= block_bytes:
                yield np.frombuffer(pending, dtype=np.float32).reshape(-1, 2).T.copy()
                pending = b""
        usable = len(pending) - len(pending) % 8
        if usable:
            yield np.frombuffer(pending[:usable], dtype=np.float32).reshape(-1, 2).T.copy()
        proc.wait()
        if proc.returncode not in (0, None):
            raise AudioLoadError(f"ffmpeg could not decode {Path(path).name}: {proc.stderr.read().decode(errors='replace').strip()}")
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.stdout.close()
        proc.stderr.close()


# --- stem storage ------------------------------------------------------------

# Stems are stored as 24-bit FLAC scaled by this factor so peaks up to +12 dBFS
# (rebuilt clipped peaks, summed stems) survive without clipping.
STORE_GAIN = 0.25


class StemWriter:
    """Append-only writer for a full-length stem file on disk."""

    def __init__(self, path: str | Path, sample_rate: int = SAMPLE_RATE):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._tmp = self.path.with_suffix(".partial.flac")
        # Fastest FLAC level: these are working files, re-read many times, never shipped.
        self._file = sf.SoundFile(str(self._tmp), "w", sample_rate, 2, "PCM_24", format="FLAC", compression_level=0.0)
        self.frames = 0
        self.clipped = 0

    def write(self, audio: np.ndarray) -> None:
        scaled = np.asarray(audio, dtype=np.float32) * STORE_GAIN
        over = np.abs(scaled) > 1.0
        if over.any():
            self.clipped += int(over.sum())
            scaled = np.clip(scaled, -1.0, 1.0)
        self._file.write(scaled.T)
        self.frames += scaled.shape[-1]

    def close(self) -> None:
        self._file.close()
        os.replace(self._tmp, self.path)  # only complete files get the final name

    def abort(self) -> None:
        self._file.close()
        self._tmp.unlink(missing_ok=True)

    def __enter__(self) -> "StemWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is None:
            self.close()
        else:
            self.abort()


def stem_frames(path: str | Path) -> int:
    return sf.info(str(path)).frames


def read_stem(path: str | Path, start: int = 0, stop: int | None = None) -> np.ndarray:
    """Read frames ``[start, stop)`` of a stored stem as float32 ``(2, n)``."""
    with sf.SoundFile(str(path)) as f:
        stop = f.frames if stop is None else min(stop, f.frames)
        start = max(0, min(start, stop))
        f.seek(start)
        data = f.read(stop - start, dtype="float32", always_2d=True)
    return np.ascontiguousarray(data.T) / STORE_GAIN


def iter_stem(path: str | Path, block_frames: int, start: int = 0, stop: int | None = None) -> Iterator[tuple[int, np.ndarray]]:
    """Yield ``(offset, block)`` pairs covering ``[start, stop)`` of a stored stem."""
    total = stem_frames(path) if stop is None else stop
    for offset in range(start, total, block_frames):
        yield offset, read_stem(path, offset, min(total, offset + block_frames))


# --- finished files -------------------------------------------------------------


def resample(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    """Very-high-quality resampling with SoX (soxr), falling back to scipy's polyphase filter."""
    if source_rate == target_rate:
        return audio
    try:
        import soxr

        return np.ascontiguousarray(soxr.resample(np.asarray(audio, dtype=np.float32).T, source_rate, target_rate, quality="VHQ").T)
    except ImportError:
        pass
    from math import gcd

    from scipy.signal import resample_poly

    g = gcd(source_rate, target_rate)
    return resample_poly(audio, target_rate // g, source_rate // g, axis=-1).astype(np.float32)


class OutputWriter:
    """Streaming writer for finished files: FLAC/WAV (16/24-bit) or 320 kbps MP3."""

    def __init__(self, path: str | Path, sample_rate: int, bit_depth: int = 24):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.sample_rate = sample_rate
        self.bit_depth = bit_depth
        self.frames = 0
        self._rng = np.random.default_rng(0)
        suffix = self.path.suffix.lower()
        self._tmp = self.path.with_name(self.path.stem + ".partial" + suffix)
        if suffix == ".mp3":
            from pedalboard.io import AudioFile

            self._mp3 = AudioFile(str(self._tmp), "w", sample_rate, 2, quality="320")
            self._file = None
        else:
            fmt = {".flac": "FLAC", ".wav": "WAV", ".aiff": "AIFF", ".aif": "AIFF"}.get(suffix)
            if fmt is None:
                raise ValueError(f"Unsupported output format: {suffix}")
            self._mp3 = None
            self._file = sf.SoundFile(str(self._tmp), "w", sample_rate, 2, f"PCM_{bit_depth}", format=fmt)

    def write(self, audio: np.ndarray) -> None:
        audio = np.asarray(audio, dtype=np.float32)
        if audio.size == 0:
            return
        if self._mp3 is not None:
            self._mp3.write(np.clip(audio, -1.0, 1.0))
        else:
            if self.bit_depth == 16:
                lsb = 1.0 / 32768.0
                audio = audio + (self._rng.random(audio.shape, dtype=np.float32) - self._rng.random(audio.shape, dtype=np.float32)) * lsb
            self._file.write(np.clip(audio, -1.0, 1.0).T)
        self.frames += audio.shape[-1]

    def close(self) -> Path:
        (self._mp3 or self._file).close()
        os.replace(self._tmp, self.path)
        return self.path

    def abort(self) -> None:
        (self._mp3 or self._file).close()
        self._tmp.unlink(missing_ok=True)

    def __enter__(self) -> "OutputWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is None:
            self.close()
        else:
            self.abort()
