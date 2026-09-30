"""Work out which song each part of the show is and find its studio original.

Matching is by melody and harmony (chroma), not by exact audio fingerprint,
because a live performance is a different recording from the studio track:
different tempo, key changes, extended intros. Two chroma sequences are
compared with a key-invariant, tempo-flexible alignment (subsequence DTW), so
a 30-second preview can be found inside a 5-minute live song.

Offline, songs are matched against your own music folder and every original
fetched before (the reference cache). Online, candidates come from Shazam,
from searching transcribed lyrics, and from iTunes / Deezer / YouTube search;
each candidate is verified by that melody match before it is accepted.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import socket
import subprocess
import sys
import tempfile
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .audio_io import SAMPLE_RATE, load_audio
from .paths import references_dir, tools_dir

log = logging.getLogger(__name__)

AUDIO_EXTENSIONS = {".mp3", ".m4a", ".flac", ".wav", ".ogg", ".opus", ".aac", ".webm", ".mp4", ".aiff", ".wma"}
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) concert-remaster"


# --- melody signatures ---------------------------------------------------------


def chroma_signature(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """CENS chroma at 2 frames per second: a compact, tempo-robust summary of melody and harmony."""
    import librosa

    mono = np.asarray(audio, dtype=np.float32)
    if mono.ndim == 2:
        mono = mono.mean(axis=0)
    y = librosa.resample(mono, orig_sr=sample_rate, target_sr=22050)
    if y.size < 22050:
        return np.zeros((12, 1), dtype=np.float32)
    return librosa.feature.chroma_cens(y=y, sr=22050, hop_length=11025, win_len_smooth=3).astype(np.float32)


def match_score(live: np.ndarray, reference: np.ndarray) -> tuple[float, int]:
    """How confidently ``reference`` is the same song as ``live``, and the key shift.

    The shorter chroma sequence is aligned inside the longer one with
    subsequence DTW, limited to tempo changes between 0.5x and 2x, once for
    every transposition. For the same song the right transposition aligns far
    better than the other eleven; for unrelated music all twelve score about
    the same. The score is that contrast: best minus median alignment quality.

    Measured on real recordings, the same song (even a different section of
    it, like a 30 s preview) scored 0.58-0.60 and unrelated music at most 0.34.
    """
    import librosa

    if live.shape[1] < 8 or reference.shape[1] < 8:
        return 0.0, 0
    steps = np.array([[1, 1], [1, 2], [2, 1]])
    scores = np.zeros(12)
    for shift in range(12):
        ref = np.roll(reference, shift, axis=0)
        query, db = (live, ref) if live.shape[1] <= ref.shape[1] else (ref, live)
        if 2 * query.shape[1] < 3 or db.shape[1] * 2 < query.shape[1]:
            return 0.0, 0
        cost = 1.0 - _centre(query).T @ _centre(db)
        acc, path = librosa.sequence.dtw(C=cost, subseq=True, step_sizes_sigma=steps, backtrack=True)
        end = int(np.argmin(acc[-1]))
        scores[shift] = 1.0 - acc[-1, end] / max(len(path), 1)
    best = int(np.argmax(scores))
    return float(np.clip(scores[best] - np.median(scores), 0.0, 1.0)), best


def _centre(chroma: np.ndarray) -> np.ndarray:
    centred = chroma - chroma.mean(axis=0, keepdims=True)
    return centred / np.maximum(np.linalg.norm(centred, axis=0, keepdims=True), 1e-9)


# --- reference library -----------------------------------------------------------


@dataclass
class Reference:
    key: str
    title: str
    artist: str
    path: str
    source: str  # library, youtube, itunes, deezer, manual
    duration: float = 0.0
    preview: bool = False  # 30-second preview rather than the full track
    url: str = ""


class ReferenceLibrary:
    """Studio originals available offline: your own music folder plus everything fetched before."""

    def __init__(self, root: Path | None = None, library_dir: str | Path | None = None):
        self.root = Path(root) if root else references_dir()
        self.root.mkdir(parents=True, exist_ok=True)
        self.index_path = self.root / "index.json"
        self.entries: dict[str, dict] = json.loads(self.index_path.read_text(encoding="utf-8")) if self.index_path.exists() else {}
        self.library_dir = Path(library_dir) if library_dir else None

    def save(self) -> None:
        self.index_path.write_text(json.dumps(self.entries, indent=2, ensure_ascii=False), encoding="utf-8")

    def add(self, ref: Reference, signature: np.ndarray | None = None) -> Reference:
        if signature is None:
            signature = chroma_signature(load_audio(ref.path))
        np.save(self.root / f"{ref.key}.sig.npy", signature)
        self.entries[ref.key] = asdict(ref)
        self.save()
        return ref

    def signature(self, key: str) -> np.ndarray | None:
        path = self.root / f"{key}.sig.npy"
        return np.load(path) if path.exists() else None

    def scan_library(self, progress=lambda message: None) -> int:
        """Index new or changed files in the user's music folder (once; cached)."""
        if not self.library_dir or not self.library_dir.is_dir():
            return 0
        added = 0
        for path in sorted(self.library_dir.rglob("*")):
            if path.suffix.lower() not in AUDIO_EXTENSIONS or not path.is_file():
                continue
            stat = path.stat()
            key = "lib-" + hashlib.sha1(f"{path}|{stat.st_size}|{int(stat.st_mtime)}".encode()).hexdigest()[:16]
            if key in self.entries:
                continue
            artist, title = _split_artist_title(path.stem)
            progress(f"Indexing {path.name}")
            try:
                self.add(Reference(key, title, artist, str(path), "library"))
                added += 1
            except Exception as exc:  # unreadable file: skip it
                log.warning("Skipping %s: %s", path, exc)
        return added

    def best_match(self, live_signature: np.ndarray, min_score: float) -> tuple[Reference | None, float]:
        best, best_score = None, 0.0
        for key, entry in self.entries.items():
            sig = self.signature(key)
            if sig is None or not Path(entry["path"]).exists():
                continue
            score, _ = match_score(live_signature, sig)
            if score > best_score:
                best, best_score = Reference(**entry), score
        return (best, best_score) if best_score >= min_score else (None, best_score)


def _split_artist_title(stem: str) -> tuple[str, str]:
    stem = re.sub(r"^\d+[\s.\-_]+", "", stem)  # "03 - " track numbers
    if " - " in stem:
        artist, title = stem.split(" - ", 1)
        return artist.strip(), title.strip()
    return "", stem.strip()


# --- online providers ----------------------------------------------------------------


def internet_available(timeout: float = 3.0) -> bool:
    for host in ("www.youtube.com", "itunes.apple.com"):
        try:
            socket.create_connection((host, 443), timeout=timeout).close()
            return True
        except OSError:
            continue
    return False


@dataclass
class Candidate:
    title: str
    artist: str
    source: str
    url: str
    duration: float = 0.0
    preview: bool = False
    score: float = 0.0
    path: str = ""
    extra: dict = field(default_factory=dict)


def _get_json(url: str, timeout: float = 15.0) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def itunes_search(query: str, limit: int = 5) -> list[Candidate]:
    url = "https://itunes.apple.com/search?" + urllib.parse.urlencode({"term": query, "entity": "song", "limit": limit})
    results = _get_json(url).get("results", [])
    return [Candidate(r.get("trackName", ""), r.get("artistName", ""), "itunes", r["previewUrl"],
                      r.get("trackTimeMillis", 0) / 1000.0, preview=True) for r in results if r.get("previewUrl")]


def deezer_search(query: str, limit: int = 5) -> list[Candidate]:
    url = "https://api.deezer.com/search?" + urllib.parse.urlencode({"q": query, "limit": limit})
    results = _get_json(url).get("data", [])
    return [Candidate(r.get("title", ""), r.get("artist", {}).get("name", ""), "deezer", r["preview"],
                      float(r.get("duration", 0)), preview=True) for r in results if r.get("preview")]


def _yt_dlp_options(cookies_browser: str = "") -> dict:
    options = {"quiet": True, "no_warnings": True, "noplaylist": True}
    if cookies_browser:
        options["cookiesfrombrowser"] = (cookies_browser,)
    deno = next(iter(sorted(tools_dir().glob("deno*/deno*"))), None)
    if deno and deno.is_file():
        options["js_runtimes"] = {"deno": {"path": str(deno)}}
    return options


def youtube_search(query: str, limit: int = 5) -> list[Candidate]:
    import yt_dlp

    with yt_dlp.YoutubeDL({**_yt_dlp_options(), "extract_flat": True}) as ydl:
        info = ydl.extract_info(f"ytsearch{limit}:{query}", download=False)
    results = []
    for entry in info.get("entries") or []:
        if not entry or not entry.get("id"):
            continue
        title = entry.get("title") or ""
        results.append(Candidate(title, entry.get("channel") or entry.get("uploader") or "", "youtube",
                                 f"https://www.youtube.com/watch?v={entry['id']}", float(entry.get("duration") or 0)))
    return results


def download(candidate: Candidate, folder: Path, cookies_browser: str = "") -> Path:
    """Fetch a candidate's audio into ``folder``; returns the file path."""
    folder.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha1(candidate.url.encode()).hexdigest()[:16]
    if candidate.source == "youtube":
        import yt_dlp

        existing = list(folder.glob(f"yt-{key}.*"))
        if existing:
            return existing[0]
        options = {**_yt_dlp_options(cookies_browser), "format": "bestaudio/best", "outtmpl": str(folder / f"yt-{key}.%(ext)s")}
        with yt_dlp.YoutubeDL(options) as ydl:
            info = ydl.extract_info(candidate.url, download=True)
            return Path(ydl.prepare_filename(info))
    suffix = ".m4a" if candidate.source == "itunes" else ".mp3"
    path = folder / f"{candidate.source}-{key}{suffix}"
    if not path.exists():
        request = urllib.request.Request(candidate.url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(request, timeout=30) as response:
            path.write_bytes(response.read())
    return path


def shazam_identify(clips: list[np.ndarray], sample_rate: int = SAMPLE_RATE) -> list[tuple[str, str]]:
    """Ask Shazam about each clip; returns (title, artist) pairs that were recognised."""
    try:
        from shazamio import Shazam
    except ImportError:
        return []
    import soundfile as sf

    async def run() -> list[tuple[str, str]]:
        shazam = Shazam()
        found = []
        for clip in clips:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                path = tmp.name
            try:
                mono = clip.mean(axis=0) if clip.ndim == 2 else clip
                sf.write(path, mono, sample_rate)
                result = await shazam.recognize(path)
                track = result.get("track")
                if track:
                    found.append((track.get("title", ""), track.get("subtitle", "")))
            except Exception as exc:
                log.info("Shazam lookup failed: %s", exc)
            finally:
                Path(path).unlink(missing_ok=True)
        return found

    try:
        return asyncio.run(run())
    except RuntimeError:  # already inside an event loop
        return []


def lyric_query(text: str, max_words: int = 12) -> str:
    """The most searchable stretch of transcribed lyrics: longest line, trimmed."""
    lines = [re.sub(r"[^\w\s']", " ", line).strip() for line in re.split(r"[.!?\n]", text)]
    lines = [line for line in lines if len(line.split()) >= 3]
    if not lines:
        return ""
    words = max(lines, key=lambda line: len(set(line.lower().split()))).split()
    return " ".join(words[:max_words])


# --- per-song identification ------------------------------------------------------------


@dataclass
class Identification:
    title: str = ""
    artist: str = ""
    score: float = 0.0
    reference: dict | None = None
    candidates: list[dict] = field(default_factory=list)
    lyrics: str = ""
    method: str = ""
    message: str = ""


def identify_song(mix: np.ndarray, vocals: np.ndarray | None, settings, library: ReferenceLibrary,
                  transcriber=None, online: bool | None = None, progress=lambda message: None) -> Identification:
    """Identify one song from its crowd-free mix (and vocal stem), and pick a studio reference."""
    result = Identification()
    signature = chroma_signature(mix)

    progress("Matching against your library and saved originals")
    ref, score = library.best_match(signature, settings.min_match_score)
    if ref is not None:
        return Identification(ref.title, ref.artist, score, asdict(ref), method=f"offline ({ref.source})",
                              message=f"Matched {ref.title} offline (score {score:.2f})")

    if online is None:
        online = settings.online and internet_available()
    if not online:
        result.message = "No offline match; online lookup is off or there is no internet"
        return result

    queries: list[str] = []
    if settings.use_shazam:
        progress("Asking Shazam")
        duration = mix.shape[-1] / SAMPLE_RATE
        clips = [mix[:, int(t * SAMPLE_RATE): int((t + 12) * SAMPLE_RATE)] for t in
                 (duration * 0.2, duration * 0.5, duration * 0.75) if t + 12 <= duration]
        for title, artist in shazam_identify(clips):
            queries.append(f"{title} {artist}")
            result.method = "shazam"
    if settings.use_lyrics_search and transcriber is not None and vocals is not None:
        progress("Transcribing lyrics")
        text = transcriber.transcribe(vocals, max_seconds=120)
        result.lyrics = text
        snippet = lyric_query(text)
        if snippet:
            queries.append(f"{settings.artist_hint} {snippet}".strip())
    if settings.artist_hint and not queries:
        queries.append(settings.artist_hint)
    queries = list(dict.fromkeys(q for q in queries if q.strip()))

    candidates: list[Candidate] = []
    for query in queries[:4]:
        progress(f"Searching: {query}")
        for search in (itunes_search, deezer_search):
            try:
                candidates.extend(search(query, limit=3))
            except Exception as exc:
                log.info("%s failed for %r: %s", search.__name__, query, exc)
        try:
            candidates.extend(c for c in youtube_search(query, limit=settings.youtube_results)
                              if 60 <= c.duration <= 12 * 60 and not _looks_like_live(c.title))
        except Exception as exc:
            log.info("YouTube search failed for %r: %s", query, exc)

    cache = library.root / "downloads"
    seen = set()
    for cand in candidates:
        if cand.url in seen:
            continue
        seen.add(cand.url)
        try:
            progress(f"Checking {cand.title} ({cand.source})")
            cand.path = str(download(cand, cache, getattr(settings, "cookies_browser", "")))
            cand.score, _ = match_score(signature, chroma_signature(load_audio(cand.path)))
        except Exception as exc:
            log.info("Could not check %s: %s", cand.url, exc)
            cand.score = 0.0
    ranked = sorted((c for c in candidates if c.path), key=lambda c: (c.score, not c.preview), reverse=True)
    result.candidates = [asdict(c) for c in ranked[:10]]
    if not ranked or ranked[0].score < settings.min_match_score:
        best = ranked[0].score if ranked else 0.0
        result.message = f"No confident match (best score {best:.2f}); treated as unreleased"
        return result

    best = ranked[0]
    # Prefer a full-length original over a 30 s preview when one matches nearly as well.
    full = next((c for c in ranked if not c.preview and c.score >= max(settings.min_match_score, best.score - 0.05)), None)
    chosen = full or best
    ref = Reference("web-" + hashlib.sha1(chosen.url.encode()).hexdigest()[:16], chosen.title, chosen.artist,
                    chosen.path, chosen.source, chosen.duration, chosen.preview, chosen.url)
    library.add(ref)
    return Identification(chosen.title, chosen.artist, chosen.score, asdict(ref), result.candidates, result.lyrics,
                          result.method or "search", f"Found {chosen.title} by {chosen.artist} ({chosen.source}, score {chosen.score:.2f})")


def _looks_like_live(title: str) -> bool:
    return bool(re.search(r"\b(live|concert|full set|tour|performance|cover|karaoke|8d|slowed|reverb|lofi|remix|mashup)\b", title, re.I))


def update_ytdlp() -> str:
    """YouTube changes often; this keeps the downloader current."""
    result = subprocess.run([sys.executable, "-m", "pip", "install", "-U", "yt-dlp[default]"], capture_output=True, text=True)
    return (result.stdout + result.stderr).strip().splitlines()[-1] if (result.stdout or result.stderr) else "done"
