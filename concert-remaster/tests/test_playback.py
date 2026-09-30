import time

import numpy as np
import pytest
import soundfile as sf
from conftest import SR, tone

from concert_remaster.audio_io import StemWriter
from concert_remaster.mixing import pan_stereo
from concert_remaster.playback import ArrayTrack, Channel, FileTrack, NullOutput, Player, Session


class ManualOutput:
    """The test pulls audio itself, like a sound card callback would."""
    rate, latency = SR, 0.0

    def __init__(self, pull):
        self.pull = pull

    def close(self):
        pass


def pull(player, frames, block=512, timeout=5.0):
    out, got, deadline = [], 0, time.time() + timeout
    while got < frames and time.time() < deadline:
        if not player.playing:
            break
        if player.queued == 0:
            time.sleep(0.001)
            continue
        n = min(block, frames - got, player.queued)
        out.append(player.output.pull(n))
        got += n
    return np.concatenate(out).T if out else np.zeros((2, 0), dtype=np.float32)


@pytest.fixture
def player():
    p = Player(output_factory=ManualOutput)
    yield p
    p.close()


def _session(tracks, start=0.0, end=None, kind="song"):
    end = end if end is not None else start + max(t.shape[-1] for t in tracks.values()) / SR
    return Session(kind, start, end, {n: Channel(ArrayTrack(a, start)) for n, a in tracks.items()})


def test_mixes_tracks_sample_locked_with_gain_and_pan(player):
    rng = np.random.default_rng(0)
    a = np.stack([tone(220, 3.0, 0.2), tone(330, 3.0, 0.2)])
    b = (rng.standard_normal((2, 3 * SR)) * 0.05).astype(np.float32)
    player.load(_session({"a": a, "b": b}))
    player.set_mix(gains={"a": 0.5, "b": 1.0}, pans={"b": -0.4})
    player.play()
    out = pull(player, SR)
    expected = a[:, : out.shape[1]] * 0.5 + pan_stereo(b, -0.4)[:, : out.shape[1]]
    assert out.shape[1] == SR
    np.testing.assert_allclose(out, expected, atol=1e-5)
    status = player.status()
    assert status["playing"] and status["position"] == pytest.approx(1.0, abs=0.01)
    assert set(status["levels"]) == {"a", "b"} and status["levels"]["a"] > -30


def test_mute_and_fader_moves_are_smooth(player):
    a = np.stack([tone(220, 4.0, 0.2)] * 2)
    player.load(_session({"a": a}))
    player.play()
    pull(player, SR // 2)
    player.set_mix(gains={"a": 0.0})
    out = pull(player, SR)
    # No click: the change is ramped, and after the queued audio the track is silent.
    assert np.max(np.abs(np.diff(out[0]))) < 0.05
    assert np.max(np.abs(out[:, -SR // 4:])) == 0.0


def test_seek_pause_and_end(player):
    a = np.stack([np.arange(2 * SR, dtype=np.float32) / (4 * SR)] * 2)  # a ramp: value tells the time
    player.load(_session({"a": a}, start=100.0))
    player.play(position=101.0)
    out = pull(player, 1000)
    assert out[0, 0] == pytest.approx(SR / (4 * SR), abs=1e-6)
    player.pause()
    assert not player.playing and player.status()["position"] == pytest.approx(101.0 + 1000 / SR, abs=1e-3)
    player.play()
    rest = pull(player, 5 * SR)
    assert rest.shape[1] == pytest.approx(SR - 1000, abs=SR * 0.01)
    assert not player.playing and player.status()["position"] == pytest.approx(100.0)  # back to the start


def test_loop_range_wraps(player):
    a = np.stack([np.arange(4 * SR, dtype=np.float32) / (8 * SR)] * 2)
    player.load(_session({"a": a}))
    player.set_loop(True, 1.0, 1.5)
    player.play(position=1.0)
    out = pull(player, int(1.2 * SR))
    # Values stay within the loop range and jump back to its start.
    t = out[0] * 8
    assert t.min() >= 1.0 - 1e-4 and t.max() < 1.5
    assert np.sum(np.diff(t) < -0.4) == 2


def test_streams_stored_stems_with_offset_and_resampling(tmp_path, player):
    a = np.stack([tone(440, 2.0, 0.3)] * 2)
    stem = tmp_path / "vocals.flac"
    with StemWriter(stem) as w:
        w.write(a)
    other = tmp_path / "mix.wav"
    sf.write(other, np.stack([tone(440, 2.0, 0.3, sr=48000)] * 2).T, 48000, subtype="FLOAT")
    session = Session("song", 50.0, 52.0, {"vocals": Channel(FileTrack(stem, offset=50.0)),
                                           "file": Channel(FileTrack(other, offset=50.0, stored_stem=False), gain=0.0)})
    player.load(session)
    player.play(position=50.5)
    out = pull(player, SR // 2)
    np.testing.assert_allclose(out, a[:, SR // 2: SR], atol=2e-4)  # 24-bit storage
    player.set_mix(gains={"vocals": 0.0, "file": 1.0})
    player.seek(50.5)
    out = pull(player, SR // 2)
    np.testing.assert_allclose(out[:, 2000:-2000], a[:, SR // 2 + 2000: SR - 2000], atol=5e-3)


def test_limiter_keeps_loud_monitoring_clean(player):
    a = np.stack([tone(100, 2.0, 0.9)] * 2)
    player.load(_session({"a": a, "b": a.copy()}))  # 1.8 peak together
    player.play()
    out = pull(player, SR)
    assert np.max(np.abs(out)) <= 1.0
    assert np.max(np.abs(out[:, SR // 2:])) == pytest.approx(10 ** (-1 / 20), abs=0.02)


def test_null_output_plays_in_real_time():
    p = Player()
    p.output_factory = lambda pull_fn: NullOutput(pull_fn)
    try:
        p.load(_session({"a": np.zeros((2, SR), dtype=np.float32)}), play=True)
        time.sleep(0.4)
        assert 0.2 < p.status()["position"] < 0.7
    finally:
        p.close()


def test_release_lets_go_of_a_projects_files(player):
    s = _session({"a": np.zeros((2, SR), dtype=np.float32)})
    s.info = {"project": "show"}
    player.load(s, play=True)
    player.release("other")
    assert player.status()["loaded"]
    player.release("show")
    status = player.status()
    assert not status["loaded"] and not status["playing"]
