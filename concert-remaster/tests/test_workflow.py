import json

import numpy as np
import pytest
import soundfile as sf
from conftest import SR

from concert_remaster.analysis import integrated_lufs
from concert_remaster.engine import Project
from concert_remaster.settings import Settings
from concert_remaster.workflow import Job, renumber


@pytest.fixture
def show_file(tmp_path, music):
    """12 s song | 4 s talk-ish | 3 s applause | 12 s song."""
    rng = np.random.default_rng(3)
    song = np.tile(music, 2)[:, : 12 * SR] * 0.5
    talk = np.stack([np.sin(2 * np.pi * 150 * np.arange(4 * SR) / SR)] * 2).astype(np.float32) * 0.1
    applause = rng.standard_normal((2, 3 * SR)).astype(np.float32) * 0.1
    audio = np.concatenate([song, talk, applause, song[:, ::-1]], axis=1)
    path = tmp_path / "My Gig.wav"
    sf.write(path, audio.T, SR, subtype="PCM_24")
    return path


def _settings():
    s = Settings()
    s.hardware.chunk_seconds = 60
    s.hardware.chunk_overlap_seconds = 2
    s.crowd.fade_seconds = 0.5
    s.identify.enabled = False
    s.speech.transcribe = False
    s.output.format = "flac"
    s.output.stems = "both"
    s.master.max_limiting_db = 12.0  # the synthetic test music is far peakier than real music
    return s


SEGMENTS = [
    {"kind": "song", "start": 0.0, "end": 12.0},
    {"kind": "talk", "start": 12.0, "end": 16.0},
    {"kind": "crowd", "start": 16.0, "end": 19.0},
    {"kind": "song", "start": 19.0, "end": 31.0},
]


def _analyzed(tmp_path, show_file, fake_backend, settings=None):
    project = Project.create(show_file, tmp_path / "projects", settings or _settings())
    job = Job(project, backend=fake_backend)
    job.analyze()
    segments = renumber([dict(s) for s in SEGMENTS])
    segments[0]["title"], segments[0]["artist"] = "Opening Song", "The Band"
    segments[1]["transcript"] = [{"start": 13.0, "end": 15.0, "text": "Thank you Mumbai!"}]
    project.update(lambda state: state.__setitem__("segments", segments))
    return project, job


def test_analyze_then_export_everything(tmp_path, show_file, fake_backend):
    project, job = _analyzed(tmp_path, show_file, fake_backend)
    assert project.read_progress()["status"] == "done"
    job.export()
    out = project.output_dir
    songs = sorted((out / "Songs").glob("*.flac"))
    assert [p.name for p in songs] == ["01 - Opening Song (The Band).flac", "02 - Song 02.flac"]
    audio, rate = sf.read(songs[0], always_2d=True)
    assert rate == 48000 and audio.shape[0] == pytest.approx(12 * 48000, abs=10)
    assert integrated_lufs(audio.T, rate) == pytest.approx(-14.0, abs=1.0)

    stems = {p.stem for p in (out / "Stems" / "01 - Opening Song (The Band)").glob("*.flac")}
    assert {"lead_vocals", "backing_vocals", "drums", "bass", "guitar", "piano", "woodwinds", "other"} <= stems
    assert {p.stem for p in (out / "Stems" / "Full Concert").glob("*.flac")} >= {"drums", "bass"}

    full = sf.info(out / "My Gig - Full Concert.flac").duration
    vibes = sf.info(out / "My Gig - Concert Vibes.flac").duration
    # Talk is kept in the full show and removed from the vibes edition.
    assert full == pytest.approx(31.0, abs=1.5)
    assert vibes == pytest.approx(31.0 - 4.0, abs=1.5)
    cue = (out / "My Gig - Full Concert.cue").read_text()
    assert 'TITLE "Opening Song"' in cue and "INDEX 01 00:00:00" in cue and "TRACK 02" in cue
    assert "Thank you Mumbai!" in (out / "My Gig - Full Concert.srt").read_text()
    report = json.loads((out / "report.json").read_text())
    assert len(report["songs"]) == 2 and report["songs"][0]["loudness_lufs"] == pytest.approx(-14.0, abs=1.0)


def test_removing_talk_and_applause_shortens_the_show(tmp_path, show_file, fake_backend):
    settings = _settings()
    settings.speech.action = "remove"
    settings.crowd.between_songs = "remove"
    settings.output.stems = "none"
    settings.output.vibes_edition = False
    project, job = _analyzed(tmp_path, show_file, fake_backend, settings)
    job.export()
    full = sf.info(project.output_dir / "My Gig - Full Concert.flac").duration
    assert full == pytest.approx(24.0, abs=1.5)
    assert not (project.output_dir / "Stems").exists()


def test_excluded_song_is_left_out(tmp_path, show_file, fake_backend):
    project, job = _analyzed(tmp_path, show_file, fake_backend)
    def drop_second(state):
        state["segments"][3]["include"] = False
    project.update(drop_second)
    job.export()
    assert len(list((project.output_dir / "Songs").glob("*.flac"))) == 1


def test_mp3_and_16_bit(tmp_path, show_file, fake_backend):
    settings = _settings()
    settings.output.format = "mp3"
    settings.output.stems = "none"
    project, job = _analyzed(tmp_path, show_file, fake_backend, settings)
    job.export()
    assert len(list((project.output_dir / "Songs").glob("*.mp3"))) == 2
