"""The app's audio engine: mixes and plays tracks through the sound card.

Everything runs in this process: a feeder thread reads each track from disk
in small blocks, applies the mixer (fader, pan, mute/solo, all smoothed),
sums, limits and meters, and queues the result; the sound card's callback
only copies queued audio out. Faders therefore act within a fraction of a
second, songs of any length stream from disk, and all tracks stay
sample-locked because they are read and summed together.

Output goes through PortAudio (``sounddevice``): WASAPI on Windows when it
can take 44.1 kHz, otherwise the default device at its own rate with
high-quality resampling. ``CONCERT_REMASTER_AUDIO=null`` plays into a silent
clocked output instead (for tests and machines without a sound card).
"""

from __future__ import annotations

import collections
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .audio_io import SAMPLE_RATE, STORE_GAIN

log = logging.getLogger(__name__)

SR = SAMPLE_RATE
BLOCK = 2048            # frames rendered per step (46 ms)
QUEUE_SECONDS = 0.2     # audio kept ready ahead of the sound card
CEILING = 10 ** (-1.0 / 20)


class AudioUnavailable(RuntimeError):
    pass


# --- track readers -------------------------------------------------------------------------


class FileTrack:
    """Streams a stored stem (or any audio file) from ``offset`` seconds of show time."""

    def __init__(self, path: str | Path, offset: float = 0.0, stored_stem: bool = True):
        import soundfile as sf

        self.file = sf.SoundFile(str(path))
        self.rate = self.file.samplerate
        self.offset = offset
        self.scale = 1.0 / STORE_GAIN if stored_stem else 1.0
        self.resampler = None
        self.pending = np.zeros((0, 2), dtype=np.float32)

    def seek(self, t: float) -> None:
        frame = int(round((t - self.offset) * self.rate))
        self.lead = max(0, -frame)  # silence before the file starts
        self.file.seek(min(max(0, frame), self.file.frames))
        self.pending = np.zeros((0, 2), dtype=np.float32)
        if self.rate != SR:
            import soxr

            self.resampler = soxr.ResampleStream(self.rate, SR, 2, dtype="float32", quality="HQ")
            self.lead = int(round(self.lead * SR / self.rate))

    def _raw(self, n: int) -> np.ndarray:
        data = self.file.read(n, dtype="float32", always_2d=True)
        if data.shape[1] == 1:
            data = np.repeat(data, 2, axis=1)
        return data[:, :2]

    def read(self, n: int) -> np.ndarray:
        out = np.zeros((2, n), dtype=np.float32)
        lead = min(self.lead, n)
        self.lead -= lead
        need = n - lead
        if need <= 0:
            return out
        if self.resampler is None:
            data = self._raw(need)
        else:
            while self.pending.shape[0] < need:
                raw = self._raw(max(1024, int(need * self.rate / SR) + 64))
                last = raw.shape[0] == 0
                chunk = self.resampler.resample_chunk(raw, last=last)
                self.pending = np.concatenate([self.pending, chunk])
                if last:
                    break
            data, self.pending = self.pending[:need], self.pending[need:]
        out[:, lead: lead + data.shape[0]] = data.T * self.scale
        return out

    def close(self) -> None:
        self.file.close()


class ArrayTrack:
    """A track held in memory (a rendered preview), starting at ``offset`` seconds."""

    def __init__(self, audio: np.ndarray, offset: float = 0.0):
        self.audio = np.asarray(audio, dtype=np.float32)
        self.offset = offset
        self.pos = 0

    def seek(self, t: float) -> None:
        self.pos = int(round((t - self.offset) * SR))

    def read(self, n: int) -> np.ndarray:
        out = np.zeros((2, n), dtype=np.float32)
        a, b = self.pos, self.pos + n
        lo, hi = max(0, a), min(self.audio.shape[-1], b)
        if hi > lo:
            out[:, lo - a: hi - a] = self.audio[:, lo:hi]
        self.pos = b
        return out

    def close(self) -> None:
        pass


# --- outputs -----------------------------------------------------------------------------------


class SoundCard:
    """PortAudio output that pulls from the player's queue."""

    def __init__(self, pull):
        try:
            import sounddevice as sd
        except (ImportError, OSError) as exc:
            raise AudioUnavailable(f"No audio output available ({exc})") from exc
        self.sd = sd
        self.pull = pull
        self.stream = None
        errors = []
        for rate, device, extra in self._candidates():
            try:
                stream = sd.OutputStream(samplerate=rate, device=device, channels=2, dtype="float32",
                                         callback=self._callback, latency="low", extra_settings=extra)
                stream.start()
                self.stream, self.rate = stream, int(rate)
                self.latency = float(stream.latency)
                log.info("Audio output: %s at %d Hz", sd.query_devices(stream.device)["name"], self.rate)
                return
            except Exception as exc:  # try the next way of opening the device
                errors.append(str(exc))
        raise AudioUnavailable("Could not open the sound card: " + "; ".join(errors[-2:]))

    def _candidates(self):
        sd = self.sd
        try:
            for index, api in enumerate(sd.query_hostapis()):
                if "WASAPI" in api["name"] and api["default_output_device"] >= 0:
                    yield SR, api["default_output_device"], sd.WasapiSettings(auto_convert=True)
        except Exception:
            pass
        yield SR, None, None
        try:
            yield int(sd.query_devices(kind="output")["default_samplerate"]), None, None
        except Exception:
            pass

    def _callback(self, outdata, frames, _time, _status):
        outdata[:] = self.pull(frames)

    def close(self) -> None:
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()


class NullOutput:
    """A silent output that consumes audio in real time, like a sound card would."""

    def __init__(self, pull, rate: int = SR, block: int = 1024):
        self.pull, self.rate, self.latency = pull, rate, 0.0
        self.running = True
        self.thread = threading.Thread(target=self._run, args=(block,), daemon=True)
        self.thread.start()

    def _run(self, block: int) -> None:
        start, played = time.perf_counter(), 0
        while self.running:
            self.pull(block)
            played += block
            delay = start + played / self.rate - time.perf_counter()
            if delay > 0:
                time.sleep(delay)

    def close(self) -> None:
        self.running = False


def open_output(pull):
    if os.environ.get("CONCERT_REMASTER_AUDIO", "").lower() == "null":
        return NullOutput(pull)
    return SoundCard(pull)


# --- the player --------------------------------------------------------------------------------


@dataclass
class Channel:
    reader: object
    gain: float = 1.0          # target linear gain (0 = muted)
    pan: float = 0.0
    current: float = 1.0       # smoothed gain actually applied
    level_db: float = -120.0


@dataclass
class Session:
    """What is loaded: one song's tracks, a clip of the show, a file or a preview."""
    kind: str
    start: float
    end: float
    channels: dict[str, Channel]
    info: dict = field(default_factory=dict)


def _pan_gains(pan: float) -> tuple[float, float, float, float]:
    """(L<-L, L<-R, R<-R, R<-L) matching the Web-standard stereo panner and the export."""
    pan = max(-1.0, min(1.0, pan))
    if abs(pan) < 1e-4:
        return 1.0, 0.0, 1.0, 0.0
    x = pan + 1.0 if pan <= 0 else pan
    gl, gr = float(np.cos(x * np.pi / 2)), float(np.sin(x * np.pi / 2))
    return (1.0, gl, gr, 0.0) if pan <= 0 else (gl, 0.0, 1.0, gr)


class Player:
    """One audio engine for the whole app. All methods are thread-safe."""

    def __init__(self, output_factory=open_output):
        self.output_factory = output_factory
        self.output = None
        self.lock = threading.RLock()
        self.session: Session | None = None
        self.playing = False
        self.render_time = 0.0       # show time of the next frame to render
        self.heard_time = 0.0        # show time of the frame at the speakers
        self.queue: collections.deque = collections.deque()
        self.queued = 0              # frames queued and not yet played
        self.head = 0                # frames of the first queued block already played
        self.master = 1.0
        self.limiter = 1.0
        self.loop = False
        self.loop_range: tuple[float, float] | None = None
        self.generation = 0
        self.resampler = None
        self.master_db = -120.0
        self.error: str | None = None
        self.stopping = False
        self.feeder = threading.Thread(target=self._feed, daemon=True, name="audio-feeder")
        self.wake = threading.Event()
        self.feeder.start()

    # --- control
    def _ensure_output(self) -> None:
        if self.output is None:
            try:
                self.output = self.output_factory(self._pull)
            except Exception:
                with self.lock:
                    self.playing = False
                raise

    def load(self, session: Session, position: float | None = None, play: bool | None = None) -> None:
        with self.lock:
            was = self.playing
            old = self.session
            self.session = session
            self._seek_locked(session.start if position is None else position)
            self.playing = was if play is None else play
            if old is not None and old is not session:
                for ch in old.channels.values():
                    ch.reader.close()
        if self.playing:
            self._ensure_output()
        self.wake.set()

    def play(self, position: float | None = None) -> None:
        with self.lock:
            if self.session is None:
                return
            if position is not None:
                self._seek_locked(position)
            elif self.heard_time >= self.session.end - 0.05:
                self._seek_locked(self.session.start)
            self.playing = True
        self._ensure_output()
        self.wake.set()

    def pause(self) -> None:
        with self.lock:
            if self.playing:
                self._seek_locked(self.heard_time)
            self.playing = False

    def stop(self) -> None:
        with self.lock:
            self.playing = False
            if self.session is not None:
                self._seek_locked(self.session.start)

    def seek(self, t: float) -> None:
        with self.lock:
            self._seek_locked(t)
        self.wake.set()

    def _seek_locked(self, t: float) -> None:
        s = self.session
        if s is None:
            return
        t = min(max(t, s.start), s.end)
        for ch in s.channels.values():
            ch.reader.seek(t)
            ch.current = ch.gain  # no fade-in from an old value after a jump
        self.render_time = self.heard_time = t
        self.queue.clear()
        self.queued = self.head = 0
        self.generation += 1
        self.resampler = None

    def set_mix(self, gains: dict[str, float] | None = None, pans: dict[str, float] | None = None,
                master: float | None = None) -> None:
        with self.lock:
            s = self.session
            if s is not None:
                for name, g in (gains or {}).items():
                    if name in s.channels:
                        ch = s.channels[name]
                        ch.gain = max(0.0, float(g))
                        if not self.playing:
                            ch.current = ch.gain  # nothing is sounding, so no need to ramp
                for name, p in (pans or {}).items():
                    if name in s.channels:
                        s.channels[name].pan = float(p)
            if master is not None:
                self.master = max(0.0, float(master))

    def set_loop(self, on: bool, start: float | None = None, end: float | None = None) -> None:
        with self.lock:
            self.loop = bool(on)
            self.loop_range = (float(start), float(end)) if start is not None and end is not None and end - start > 0.2 else None

    def status(self) -> dict:
        with self.lock:
            s = self.session
            return {
                "loaded": s is not None,
                "kind": s.kind if s else None,
                "info": s.info if s else {},
                "start": s.start if s else 0.0,
                "end": s.end if s else 0.0,
                "playing": self.playing,
                "position": round(self.heard_time, 3),
                "levels": {n: round(c.level_db, 1) for n, c in s.channels.items()} if s and self.playing else {},
                "master_level": round(self.master_db, 1) if self.playing else -120.0,
                "loop": self.loop,
                "loop_range": self.loop_range,
                "buffering": self.playing and self.queued == 0,
                "error": self.error,
            }

    def release(self, project: str) -> None:
        """Stop and let go of a project's files (Windows can't rewrite files that are open)."""
        with self.lock:
            s = self.session
            if s is None or s.info.get("project") != project:
                return
            self.playing = False
            self.session = None
            self.queue.clear()
            self.queued = self.head = 0
            for ch in s.channels.values():
                ch.reader.close()

    def close(self) -> None:
        self.stopping = True
        self.wake.set()
        with self.lock:
            self.playing = False
            if self.session is not None:
                for ch in self.session.channels.values():
                    ch.reader.close()
            self.session = None
        if self.output is not None:
            self.output.close()
            self.output = None

    # --- audio thread: hand queued audio to the sound card
    def _pull(self, frames: int) -> np.ndarray:
        out = np.zeros((frames, 2), dtype=np.float32)
        with self.lock:
            filled = 0
            while filled < frames and self.queue:
                gen, data, t0, rate, levels, master_db = self.queue[0]
                take = min(frames - filled, data.shape[0] - self.head)
                out[filled: filled + take] = data[self.head: self.head + take]
                filled += take
                self.head += take
                self.queued -= take
                self.heard_time = t0 + self.head / rate
                if levels is not None and self.session is not None:
                    for name, db in levels.items():
                        if name in self.session.channels:
                            self.session.channels[name].level_db = db
                    self.master_db = master_db
                if self.head >= data.shape[0]:
                    self.queue.popleft()
                    self.head = 0
            if self.playing and not self.queue and self.session and self.render_time >= self.session.end:
                self.playing = False  # reached the end
                self._seek_locked(self.session.start)
        self.wake.set()
        return out

    # --- feeder thread: read, mix and queue
    def _feed(self) -> None:
        while not self.stopping:
            self.wake.wait(0.02)
            self.wake.clear()
            while not self.stopping:
                with self.lock:
                    s = self.session
                    rate = getattr(self.output, "rate", SR)
                    if not (self.playing and s is not None and self.queued < QUEUE_SECONDS * rate
                            and self.render_time < s.end):
                        break
                    try:
                        self._render_block(s, rate)
                    except Exception as exc:  # a broken file must not kill the engine
                        log.exception("Playback failed")
                        self.error = f"{type(exc).__name__}: {exc}"
                        self.playing = False
                        break

    def _render_block(self, s: Session, rate: int) -> None:
        if self.loop:
            lo, hi = self.loop_range or (s.start, s.end)
            if self.render_time >= hi - 1e-6:
                t = lo
                for ch in s.channels.values():
                    ch.reader.seek(t)
                self.render_time = t
        hi = self.loop_range[1] if self.loop and self.loop_range else s.end
        n = max(1, min(BLOCK, int(round((hi - self.render_time) * SR))))
        mix = np.zeros((2, n), dtype=np.float32)
        ramp = np.linspace(0.0, 1.0, n, endpoint=False, dtype=np.float32)
        levels = {}
        for name, ch in s.channels.items():
            x = ch.reader.read(n)
            if ch.gain == 0.0 and ch.current == 0.0:
                ch_level = -120.0
            else:
                g = ch.current + (ch.gain - ch.current) * ramp
                ll, lr, rr, rl = _pan_gains(ch.pan)
                y0 = (x[0] * ll + x[1] * lr) * g
                y1 = (x[1] * rr + x[0] * rl) * g
                mix[0] += y0
                mix[1] += y1
                peak = float(max(np.abs(y0).max(initial=0.0), np.abs(y1).max(initial=0.0)))
                ch_level = 20 * np.log10(peak * self.master + 1e-9)
            ch.current = ch.gain
            levels[name] = ch_level
        mix *= self.master
        # A simple peak limiter so loud monitoring never distorts (the export has real mastering).
        peak = float(np.abs(mix).max(initial=0.0))
        target = min(1.0, CEILING / peak) if peak > 0 else 1.0
        if target < self.limiter:
            gains = np.full(n, target, dtype=np.float32)
        else:
            release = min(target, self.limiter * 10 ** (6.0 * n / SR / 20))  # recover 6 dB/s
            gains = self.limiter + (release - self.limiter) * ramp
            target = release
        mix *= gains
        self.limiter = target
        np.clip(mix, -1.0, 1.0, out=mix)
        master_db = 20 * np.log10(float(np.abs(mix).max(initial=0.0)) + 1e-9)
        out = mix.T
        if rate != SR:
            import soxr

            if self.resampler is None:
                self.resampler = soxr.ResampleStream(SR, rate, 2, dtype="float32", quality="HQ")
            out = self.resampler.resample_chunk(np.ascontiguousarray(out))
        t0 = self.render_time - getattr(self.output, "latency", 0.0)
        self.queue.append((self.generation, np.ascontiguousarray(out, dtype=np.float32), t0, rate, levels, master_db))
        self.queued += out.shape[0]
        self.render_time += n / SR


_player: Player | None = None
_player_lock = threading.Lock()


def get_player() -> Player:
    global _player
    with _player_lock:
        if _player is None:
            _player = Player()
        return _player
