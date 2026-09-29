"""Automatic stem balancing and summing.

Live mixes heard from the audience usually have the vocal buried, the bass
boomy and the cymbals washing over everything. The auto-mixer measures each
stem's loudness and nudges it toward a studio-style balance relative to the
lead vocal, only partway (``strength``) and never by more than
``max_adjust_db``, so the band's own dynamics and arrangement survive.

Stems that are far quieter than the rest are treated as "ghosts": the model
found nothing real there (a piano stem for a song without piano), so what it
holds is separation bleed. Those are turned down rather than boosted.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .analysis import db_to_gain, integrated_lufs
from .stem_processing import StemProfile, profile_for


@dataclass
class BalanceResult:
    gains_db: dict[str, float]
    loudness_lufs: dict[str, float]
    ghosts: list[str]
    reference: str | None


def auto_balance(
    stems: dict[str, np.ndarray],
    sample_rate: int,
    strength: float = 0.6,
    max_adjust_db: float = 6.0,
    ghost_threshold_lu: float = 30.0,
    ghost_cut_db: float = -6.0,
    user_gains_db: dict[str, float] | None = None,
    profiles: dict[str, StemProfile] | None = None,
) -> BalanceResult:
    user_gains_db = user_gains_db or {}
    loudness = {name: integrated_lufs(audio, sample_rate) for name, audio in stems.items()}
    finite = {name: value for name, value in loudness.items() if np.isfinite(value)}
    if not finite:
        return BalanceResult({name: user_gains_db.get(name, 0.0) for name in stems}, loudness, [], None)

    loudest = max(finite.values())
    ghosts = sorted(name for name in stems if name not in finite or finite[name] < loudest - ghost_threshold_lu)
    audible = {name: value for name, value in finite.items() if name not in ghosts}
    reference = "vocals" if "vocals" in audible else max(audible, key=audible.get)
    reference_target = profile_for(reference, profiles).balance_db

    gains: dict[str, float] = {}
    for name in stems:
        if name in ghosts:
            gain = ghost_cut_db
        else:
            current = loudness[name] - loudness[reference]
            desired = profile_for(name, profiles).balance_db - reference_target
            gain = float(np.clip((desired - current) * strength, -max_adjust_db, max_adjust_db))
        gains[name] = gain + user_gains_db.get(name, 0.0)
    return BalanceResult(gains, loudness, ghosts, reference)


def sum_stems(stems: dict[str, np.ndarray], gains_db: dict[str, float]) -> np.ndarray:
    if not stems:
        raise ValueError("No stems to mix")
    length = max(audio.shape[-1] for audio in stems.values())
    mix = np.zeros((2, length), dtype=np.float32)
    for name, audio in stems.items():
        mix[:, : audio.shape[-1]] += audio * np.float32(db_to_gain(gains_db.get(name, 0.0)))
    return mix
