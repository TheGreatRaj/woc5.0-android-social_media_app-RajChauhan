import numpy as np
import pytest

SR = 44100


def tone(freq, seconds=2.0, amp=0.3, sr=SR):
    t = np.arange(int(seconds * sr)) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def stereo(mono):
    return np.stack([mono, mono]).astype(np.float32)


@pytest.fixture
def music():
    """Six seconds of band-like audio: bass, chords, a 'vocal' line and drum hits."""
    rng = np.random.default_rng(1)
    seconds = 6.0
    n = int(seconds * SR)
    t = np.arange(n) / SR
    bass = 0.25 * np.sin(2 * np.pi * 55 * t)
    chords = sum(0.08 * np.sin(2 * np.pi * f * t) for f in (220, 277, 330))
    vocal = 0.2 * np.sin(2 * np.pi * (440 + 30 * np.sin(2 * np.pi * 5 * t)) * t) * (np.sin(2 * np.pi * 0.5 * t) > 0)
    hits = np.zeros(n)
    for start in range(0, n, SR // 2):
        length = min(4000, n - start)
        hits[start : start + length] += rng.standard_normal(length) * np.exp(-np.arange(length) / 600) * 0.4
    left = bass + chords + vocal + hits
    right = bass + 0.8 * chords + vocal + hits
    return np.stack([left, right]).astype(np.float32)


class FakeBackend:
    """Stands in for the AI models with fixed proportional splits, so stems always sum to the input."""

    SPLITS = [
        ("crowd", {"crowd": 0.1, "other": 0.9}),
        ("dereverb", {"noreverb": 0.8, "reverb": 0.2}),
        ("echo", {"noreverb": 0.8, "reverb": 0.2}),
        ("denoise", {"dry": 0.9, "other": 0.1}),
        ("karaoke", {"vocals": 0.7, "instrumental": 0.3}),
        ("drumsep", {"kick": 0.3, "snare": 0.25, "toms": 0.1, "hh": 0.15, "ride": 0.1, "crash": 0.1}),
        ("wind", {"woodwinds": 0.2, "nowoodwinds": 0.8}),
        ("-sw", {"vocals": 0.05, "drums": 0.3, "bass": 0.25, "guitar": 0.2, "piano": 0.1, "other": 0.1}),
        ("6s", {"vocals": 0.05, "drums": 0.3, "bass": 0.25, "guitar": 0.2, "piano": 0.1, "other": 0.1}),
        ("demucs", {"vocals": 0.25, "drums": 0.3, "bass": 0.25, "other": 0.2}),
    ]

    def __init__(self):
        self.calls = []

    def separate(self, audio, sample_rate, model):
        self.calls.append(model)
        name = model.lower()
        for token, parts in self.SPLITS:
            if token in name:
                return {stem: audio * share for stem, share in parts.items()}
        return {"vocals": audio * 0.4, "instrumental": audio * 0.6}


@pytest.fixture
def fake_backend():
    return FakeBackend()
