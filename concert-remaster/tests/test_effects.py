import json

import numpy as np
import pytest
import scipy.signal as ss
from conftest import SR
from sfx import applause, cheering, co2_jet, confetti, firework, place
from test_workflow import _analyzed, _settings, show_file  # noqa: F401 (fixture)

from concert_remaster.audio_io import StemWriter
from concert_remaster.effects import EffectEvent, detect_effects, detect_in_file, repair, ride_levels


def _audience(seconds, rng):
    crowd = place(seconds, [(0, applause(seconds, 0.03, rng)), (0, cheering(seconds, 0.02, rng))])
    return crowd


def _kinds_near(events, t, tolerance=1.0):
    return {e.kind for e in events if e.start - tolerance <= t <= e.end + tolerance}


def test_finds_each_effect_over_the_audience():
    rng = np.random.default_rng(0)
    crowd = _audience(40, rng) + place(40, [(8, co2_jet(2.0, 0.3, rng)), (18, firework(2.5, 0.3, rng)),
                                            (29, confetti(1.5, 0.3, rng))])
    events = detect_effects(crowd, SR)
    assert "co2" in _kinds_near(events, 9.0)
    assert "firework" in _kinds_near(events, 18.2)
    assert "confetti" in _kinds_near(events, 29.1)
    # Nothing is reported where there is only the audience.
    assert all(_kinds_near([e], t, 0.5) == set() for e in events for t in (3.0, 13.5, 24.0, 36.0))


@pytest.mark.parametrize("make", [applause, cheering])
def test_audience_alone_is_not_an_effect(make):
    rng = np.random.default_rng(5)
    # A loud cheer rising out of a quiet crowd is the classic false alarm.
    quiet = make(10, 0.01, rng)
    loud = make(10, 0.2, rng)
    crowd = place(30, [(0, quiet), (10, loud), (20, quiet)])
    assert detect_effects(crowd, SR) == []


def test_detection_on_a_stored_stem_matches_block_boundaries(tmp_path):
    rng = np.random.default_rng(1)
    crowd = _audience(50, rng) + place(50, [(19.5, co2_jet(2.0, 0.3, rng)), (35, firework(2.5, 0.3, rng))])
    path = tmp_path / "crowd.flac"
    with StemWriter(path) as writer:
        writer.write(crowd)
    seen = []
    events = detect_in_file(path, block_s=20.0, overlap_s=5.0, progress=seen.append)
    # The jet straddles the 20 s block boundary and must be found once, whole.
    jets = [e for e in events if e.kind == "co2"]
    assert len(jets) == 1 and jets[0].start < 19.7 and jets[0].end > 21.3
    assert any(e.kind == "firework" for e in events)
    assert seen[-1] == 1.0
    assert [e.kind for e in detect_in_file(path, kinds={"firework"})] == ["firework"]


def _song(seconds, rng):
    t = np.arange(int(seconds * SR)) / SR
    chords = sum(0.1 * np.sin(2 * np.pi * f * t) for f in (220, 277, 330, 440))
    hats = ss.sosfilt(ss.butter(2, 6000, "highpass", fs=SR, output="sos"), rng.standard_normal(t.size)) * 0.02
    hats *= (np.sin(2 * np.pi * 4 * t) > 0.6)
    return np.stack([chords + hats] * 2).astype(np.float32)


def test_repair_removes_the_leak_and_keeps_the_music():
    rng = np.random.default_rng(2)
    seconds = 16
    music = _song(seconds, rng)
    jet = place(seconds, [(6, co2_jet(3.0, 0.3, rng))])
    crowd = _audience(seconds, rng) + jet
    # Separation leaves a filtered part of the jet in the music stem.
    leak = ss.sosfilt(ss.butter(2, 2000, "highpass", fs=SR, output="sos"), jet, axis=-1).astype(np.float32) * 0.5
    stem = music + leak
    event = EffectEvent("co2", 5.9, 9.3, 20.0)
    fixed = repair(stem, SR, [event], reference=crowd)

    span = slice(int(6.2 * SR), int(8.8 * SR))

    def err_db(x):
        return 10 * np.log10(np.mean((x[:, span] - music[:, span]) ** 2) / np.mean(music[:, span] ** 2))

    assert err_db(fixed) < err_db(stem) - 8
    # The music keeps its level through the event (no dip) and nothing changes outside it.
    level = 10 * np.log10(np.mean(fixed[:, span] ** 2) / np.mean(music[:, span] ** 2))
    assert abs(level) < 1.5
    outside = slice(0, int(5.5 * SR))
    assert np.max(np.abs(fixed[:, outside] - stem[:, outside])) < 1e-3


def test_repair_amounts():
    rng = np.random.default_rng(3)
    music = _song(12, rng)
    burst = place(12, [(5, co2_jet(2.0, 0.4, rng))])
    stem = music + burst
    event = EffectEvent("co2", 4.9, 7.3, 20.0)
    span = slice(int(5.2 * SR), int(6.8 * SR))

    def residue(x):
        return np.mean((x[:, span] - music[:, span]) ** 2)

    assert repair(stem, SR, [event], amount=0.0) is stem
    half = residue(repair(stem, SR, [event], amount=0.5))
    full = residue(repair(stem, SR, [event], amount=1.0))
    assert full < half < residue(stem)


def test_level_riding_fills_a_dip_but_keeps_the_song_shape():
    rng = np.random.default_rng(4)
    seconds = 90
    music = _song(seconds, rng)
    envelope = np.ones(music.shape[-1], dtype=np.float32)
    envelope[int(40 * SR):int(48 * SR)] = 10 ** (-6 / 20)  # a phone's auto-gain pumping down 6 dB
    envelope[int(70 * SR):] = 10 ** (6 / 20)  # the drop: a lasting change the riding must leave alone
    audio = music * envelope

    rode, largest = ride_levels(audio, SR, range_db=4.0)

    def level(x, a, b):
        return 10 * np.log10(np.mean(x[:, int(a * SR):int(b * SR)] ** 2))

    dip_before = level(audio, 30, 38) - level(audio, 42, 46)
    dip_after = level(rode, 30, 38) - level(rode, 42, 46)
    assert dip_before == pytest.approx(6, abs=0.3)
    assert dip_after < dip_before - 2
    assert 2 < largest <= 4.0
    # The song keeps its long-term shape.
    assert level(rode, 78, 88) - level(rode, 10, 30) > 5


def test_export_cleans_effects_inside_songs(tmp_path, show_file, fake_backend):  # noqa: F811
    settings = _settings()
    settings.output.stems = "none"
    project, job = _analyzed(tmp_path, show_file, fake_backend, settings)
    events = [{"kind": "co2", "start": 3.0, "end": 5.0, "strength_db": 18.0},
              {"kind": "firework", "start": 16.5, "end": 18.0, "strength_db": 15.0}]
    project.update(lambda state: state.__setitem__("effects", events))
    job.export()
    report = json.loads((project.output_dir / "report.json").read_text())
    cleaned = report["songs"][0]["effects_cleaned"]
    assert [e["kind"] for e in cleaned] == ["co2"] and cleaned[0]["start"] == pytest.approx(3.0)
    assert report["songs"][1]["effects_cleaned"] == []
    assert "level_riding_db" in report["songs"][0]

    settings.effects.action = "keep"
    project.set_settings(settings)
    job.export()
    report = json.loads((project.output_dir / "report.json").read_text())
    assert report["songs"][0]["effects_cleaned"] == []


def test_analysis_stores_detected_effects(tmp_path, show_file, fake_backend):  # noqa: F811
    project, job = _analyzed(tmp_path, show_file, fake_backend)
    assert isinstance(project.state.get("effects"), list)
    settings = project.settings
    settings.effects.co2 = settings.effects.fireworks = settings.effects.confetti = False
    project.set_settings(settings)
    assert job.detect_effects() == [] and project.state["effects"] == []


def test_repair_leaves_music_alone_when_the_crowd_stem_has_music_bleed():
    # Real crowd stems always carry some of the music; a (mis)detected event over plain
    # music must not cancel the music itself.
    rng = np.random.default_rng(6)
    seconds = 16
    music = _song(seconds, rng)
    crowd = _audience(seconds, rng) * 0.3 + music * 0.08
    event = EffectEvent("co2", 6.0, 9.0, 12.0)
    fixed = repair(music, SR, [event], reference=crowd)
    span = slice(int(6.2 * SR), int(8.8 * SR))
    level = 10 * np.log10(np.mean(fixed[:, span] ** 2) / np.mean(music[:, span] ** 2))
    assert abs(level) < 1.0
    err = 10 * np.log10(np.mean((fixed[:, span] - music[:, span]) ** 2) / np.mean(music[:, span] ** 2))
    assert err < -15


def test_repair_still_removes_a_jet_when_the_crowd_stem_has_music_bleed():
    rng = np.random.default_rng(7)
    seconds = 16
    music = _song(seconds, rng)
    jet = place(seconds, [(6, co2_jet(3.0, 0.3, rng))])
    crowd = _audience(seconds, rng) + jet + music * 0.08
    leak = ss.sosfilt(ss.butter(2, 2000, "highpass", fs=SR, output="sos"), jet, axis=-1).astype(np.float32) * 0.5
    stem = music + leak
    fixed = repair(stem, SR, [EffectEvent("co2", 5.9, 9.3, 20.0)], reference=crowd)
    span = slice(int(6.2 * SR), int(8.8 * SR))

    def err_db(x):
        return 10 * np.log10(np.mean((x[:, span] - music[:, span]) ** 2) / np.mean(music[:, span] ** 2))

    assert err_db(fixed) < err_db(stem) - 8
    assert abs(10 * np.log10(np.mean(fixed[:, span] ** 2) / np.mean(music[:, span] ** 2))) < 1.5


def _kick_track(seconds, rng):
    t = np.arange(int(seconds * SR)) / SR
    kick = np.zeros(t.size)
    for start in range(0, t.size - SR // 4, SR // 2):  # four-on-the-floor at 120 bpm
        n = SR // 4
        kick[start:start + n] += np.sin(2 * np.pi * 55 * np.arange(n) / SR) * np.exp(-np.arange(n) / (0.12 * SR)) * 0.8
    bass = 0.15 * np.sin(2 * np.pi * 41 * t) * (np.sin(2 * np.pi * 0.25 * t) > 0)
    return np.stack([kick + bass] * 2).astype(np.float32)


def test_music_bleed_in_the_crowd_stem_is_not_an_effect():
    rng = np.random.default_rng(8)
    music = _kick_track(40, rng) + _song(40, rng)
    crowd = _audience(40, rng) * 0.3 + music * 0.1  # kick and bass bleed into the crowd stem
    assert detect_effects(crowd, SR, music=music) == []
    # ... while a real firework on top still stands out from the music.
    boom = place(40, [(20, firework(2.5, 0.3, rng))])
    found = detect_effects(crowd + boom, SR, music=music)
    assert "firework" in _kinds_near(found, 20.2)
