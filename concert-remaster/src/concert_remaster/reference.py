"""Use the studio original as a guide: tone per instrument and the mix balance.

The original is separated into stems once (with the same instrument model as
the concert) and cached next to it. Each live stem is then EQ-matched to its
studio counterpart, and the relative levels of the studio stems become the
auto-mixer's targets. Nothing from the studio recording is mixed into the
output: the performance stays 100% live.
"""

from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .analysis import integrated_lufs
from .audio_io import SAMPLE_RATE, StemWriter, load_audio, read_stem
from .separation import SeparationBackend, _INSTRUMENTS, _VOCALS
from .settings import Settings
from .stem_processing import StemProfile, profile_for

log = logging.getLogger(__name__)

# Which studio stem each live stem is compared with.
REFERENCE_STEM = {
    "vocals": "vocals",
    "lead_vocals": "vocals",
    "drums": "drums",
    "bass": "bass",
    "guitar": "guitar",
    "piano": "piano",
    "other": "other",
    "woodwinds": "other",
}


@dataclass
class ReferenceGuide:
    title: str
    mix: np.ndarray
    stems: dict[str, np.ndarray]
    preview: bool = False

    def stem_for(self, live_name: str) -> np.ndarray | None:
        name = REFERENCE_STEM.get(live_name)
        return self.stems.get(name) if name else None


def load_guide(reference: dict, backend: SeparationBackend | None, instrument_model: str, per_stem: bool) -> ReferenceGuide:
    """Load a reference (``identify.Reference`` as a dict) and its stems, separating them on first use."""
    path = Path(reference["path"])
    mix = load_audio(path)
    stems: dict[str, np.ndarray] = {}
    if per_stem and backend is not None:
        cache = path.parent / f"{path.stem}.stems"
        names = ("vocals", "drums", "bass", "guitar", "piano", "other")
        tag = cache / f"{instrument_model}.done"
        if tag.exists():
            stems = {n: read_stem(cache / f"{n}.flac") for n in names if (cache / f"{n}.flac").exists()}
        else:
            try:
                outputs = backend.separate(mix, SAMPLE_RATE, instrument_model)
                for key, audio in outputs.items():
                    name = "vocals" if key in _VOCALS else _INSTRUMENTS.get(key, "other_all").replace("other_all", "other")
                    stems[name] = stems[name] + audio if name in stems else audio
                cache.mkdir(parents=True, exist_ok=True)
                for name, audio in stems.items():
                    with StemWriter(cache / f"{name}.flac") as writer:
                        writer.write(audio)
                tag.write_text("ok")
            except Exception as exc:
                log.warning("Could not separate the reference %s: %s; matching the whole mix only", path.name, exc)
    return ReferenceGuide(reference.get("title", path.stem), mix, stems, bool(reference.get("preview")))


def balance_targets(guide: ReferenceGuide, live_names: list[str], settings: Settings) -> dict[str, StemProfile]:
    """Per-stem profiles whose ``balance_db`` copies the studio mix's relative levels."""
    loudness = {name: integrated_lufs(audio, SAMPLE_RATE) for name, audio in guide.stems.items()}
    finite = {k: v for k, v in loudness.items() if np.isfinite(v)}
    profiles = {name: profile_for(name, settings.stems) for name in live_names}
    if not finite:
        return profiles
    loudest = max(finite.values())
    vocals = finite.get("vocals")
    reference_level = vocals if vocals is not None and vocals > loudest - 20 else loudest
    for name in live_names:
        ref_name = REFERENCE_STEM.get(name)
        if ref_name in finite and finite[ref_name] > loudest - 30:
            profiles[name] = dataclasses.replace(profiles[name], balance_db=round(finite[ref_name] - reference_level, 2))
    return profiles
