import numpy as np
import pytest
from conftest import SR

from concert_remaster.audio_io import StemWriter
from concert_remaster.segmentation import compute_features, novelty_curve, segment
from concert_remaster.settings import SegmentationSettings

rng = np.random.default_rng(11)


def _band(seconds, chord):
    t = np.arange(int(seconds * SR)) / SR
    x = sum(0.08 * np.sin(2 * np.pi * f * t) for f in chord) + 0.2 * np.sin(2 * np.pi * chord[0] / 4 * t)
    hits = np.zeros_like(t)
    for s in range(0, t.size - 3000, SR // 2):
        hits[s : s + 3000] += rng.standard_normal(3000) * np.exp(-np.arange(3000) / 500) * 0.3
    return np.stack([x + hits, x + hits]).astype(np.float32)


def _sung(seconds):
    """Held notes with vibrato, changing every second: singing."""
    t = np.arange(int(seconds * SR)) / SR
    notes = 220 * 2 ** (rng.integers(0, 12, int(seconds) + 1) / 12)
    f = notes[t.astype(int)] * (1 + 0.004 * np.sin(2 * np.pi * 5.5 * t))
    phase = 2 * np.pi * np.cumsum(f) / SR
    v = 0.3 * (np.sin(phase) + 0.4 * np.sin(2 * phase))
    return np.stack([v, v]).astype(np.float32)


def _spoken(seconds):
    """Constantly gliding pitch with syllable-rate bursts: speech."""
    t = np.arange(int(seconds * SR)) / SR
    glide = np.cumsum(rng.standard_normal(t.size)) / np.sqrt(SR) * 40
    f = 140 + 30 * np.sin(2 * np.pi * 2.3 * t) + glide
    phase = 2 * np.pi * np.cumsum(np.clip(f, 80, 300)) / SR
    syllables = (np.sin(2 * np.pi * 4.5 * t) > -0.2).astype(float)
    v = 0.3 * (np.sin(phase) + 0.5 * np.sin(2 * phase)) * syllables
    return np.stack([v, v]).astype(np.float32)


def _crowd(seconds, level=0.15):
    return (rng.standard_normal((2, int(seconds * SR))) * level).astype(np.float32)


def _silence(seconds):
    return np.zeros((2, int(seconds * SR)), np.float32)


def _write(tmp_path, name, parts):
    path = tmp_path / f"{name}.flac"
    with StemWriter(path) as w:
        for p in parts:
            w.write(p)
    return path


@pytest.fixture(scope="module")
def show(tmp_path_factory):
    """song A (45 s) | applause (6 s) | talk (12 s) | song B (45 s) | applause (5 s)"""
    tmp = tmp_path_factory.mktemp("show")
    band = _write(tmp, "band", [_band(45, (220, 277, 330)), _silence(6), _band(12, (220, 277, 330)) * 0.01,
                                _band(45, (196, 247, 294)), _silence(5)])
    vocal = _write(tmp, "vocal", [_sung(45), _silence(6), _spoken(12), _sung(45), _silence(5)])
    crowd = _write(tmp, "crowd", [_crowd(45, 0.01), _crowd(6), _crowd(12, 0.02), _crowd(45, 0.01), _crowd(5)])
    return compute_features(band, vocal, crowd)


def test_finds_songs_talk_and_applause(show):
    settings = SegmentationSettings(min_song_seconds=20)
    segments = segment(show, settings, duration=113.0)
    kinds = [s["kind"] for s in segments]
    assert kinds == ["song", "crowd", "talk", "song", "crowd"], segments
    songs = [s for s in segments if s["kind"] == "song"]
    assert songs[0]["start"] == 0 and songs[0]["end"] == pytest.approx(45, abs=1.5)
    assert songs[1]["start"] == pytest.approx(63, abs=1.5)
    talk = next(s for s in segments if s["kind"] == "talk")
    assert talk["start"] == pytest.approx(51, abs=2) and talk["end"] == pytest.approx(63, abs=2)
    assert [s["track"] for s in songs] == [1, 2] and songs[0]["title"] == "Song 01"
    assert segments[-1]["end"] == 113.0
    for a, b in zip(segments, segments[1:]):
        assert a["end"] == b["start"]


def test_singing_is_not_mistaken_for_talk(show):
    hold = show.hold
    fps = 2
    sung = hold[5 * fps : 40 * fps]
    spoken = hold[53 * fps : 61 * fps]
    assert np.nanmedian(sung) > 0.5 > 0.3 > np.nanmedian(spoken)


def test_single_mode_keeps_one_song(show):
    segments = segment(show, SegmentationSettings(split_songs=False), duration=113.0)
    assert [s["kind"] for s in segments] == ["song"]


def test_continuous_set_splits_at_musical_changes(tmp_path):
    # Three 60 s "tracks" in different keys with no gaps, like a DJ set.
    band = _write(tmp_path, "band", [_band(60, (220, 277, 330)), _band(60, (185, 233, 277)), _band(60, (247, 311, 370))])
    features = compute_features(band, None, None)
    segments = segment(features, SegmentationSettings(mode="continuous", min_song_seconds=30), duration=180.0)
    songs = [s for s in segments if s["kind"] == "song"]
    assert len(songs) == 3, segments
    assert songs[1]["start"] == pytest.approx(60, abs=4) and songs[2]["start"] == pytest.approx(120, abs=4)
    curve = novelty_curve(features, 0, features.frames)
    assert curve.argmax() in range(100, 140) or curve.argmax() in range(220, 260)


def test_split_by_identity_follows_the_tracks():
    from concert_remaster.segmentation import split_by_identity

    labels = ["A", "A", "A", None, "B", "B", "X", "B", "B", "C", "C", "C", None]
    windows = [{"start": i * 10.0, "end": i * 10.0 + 20, "key": k, "title": k or "", "artist": "",
                "score": 0.7 if k else 0.1, "reference": {"key": k} if k else None} for i, k in enumerate(labels)]
    songs = split_by_identity(0.0, 140.0, windows, min_song_seconds=15)
    assert [s["title"] for s in songs] == ["A", "B", "C"]
    assert [s["start"] for s in songs] == [0.0, 45.0, 95.0] and songs[-1]["end"] == 140.0


def test_split_by_identity_gives_up_when_little_is_known():
    from concert_remaster.segmentation import split_by_identity

    windows = [{"start": i * 10.0, "end": i * 10.0 + 20, "key": None, "title": "", "artist": "", "score": 0.0, "reference": None}
               for i in range(10)]
    assert split_by_identity(0.0, 110.0, windows, 15) is None


def _windows(labels, hop=10.0, window=20.0):
    out = []
    for i, label in enumerate(labels):
        key, score = (label if isinstance(label, tuple) else (label, 0.6)) if label else (None, 0.0)
        out.append({"start": i * hop, "end": i * hop + window, "key": key, "title": key.upper() if key else "",
                    "artist": "DJ" if key else "", "score": score, "reference": {"key": key} if key else None})
    return out


def test_confident_spans_ignore_lone_weak_matches():
    from concert_remaster.segmentation import confident_spans

    ws = _windows(["a", "a", None, "a", None, None, None, ("x", 0.5), None, ("y", 0.8), "b", "b", "b"])
    spans = confident_spans(ws)
    assert [(s["key"], s["count"]) for s in spans] == [("a", 3), ("y", 1), ("b", 3)]


def test_partial_identification_names_what_it_can_and_keeps_the_rest():
    from concert_remaster.segmentation import label_by_identity

    # 0-60 s track a, 60-150 s unknown, 150-240 s track b. Novelty found a (wrong) cut at 30 s
    # inside a, and one at 100 s in the unknown part.
    labels = ["a"] * 5 + [None] * 10 + ["b"] * 8
    songs = label_by_identity(0.0, 240.0, _windows(labels), [30.0, 100.0], min_song_seconds=40)
    # a ends ~10 s after it was last heard; the unknown stretch keeps the novelty cut at 100 s.
    assert [s["title"] for s in songs] == ["A", "", "", "B"]
    assert songs[0]["start"] == 0 and songs[0]["end"] == 60        # the cut at 30 s inside a is gone
    assert songs[1]["end"] == 100 and songs[3]["start"] == 150 and songs[3]["end"] == 240
    assert songs[3]["identification"]["method"] == "set windows"
    assert label_by_identity(0.0, 100.0, _windows([None] * 9), [], 40) is None
