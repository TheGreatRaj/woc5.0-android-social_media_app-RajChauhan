import numpy as np
import pytest
import soundfile as sf
from conftest import SR

from concert_remaster.audio_io import read_stem
from concert_remaster.engine import Cancelled, Project, chunk_starts, prepare_source, run_separation
from concert_remaster.settings import Settings


def _recording(tmp_path, music, seconds=20.0, clip=True):
    reps = int(np.ceil(seconds * SR / music.shape[1]))
    audio = np.tile(music, reps)[:, : int(seconds * SR)]
    if clip:
        audio = np.clip(audio * 3.0, -1, 1)
    path = tmp_path / "gig.wav"
    sf.write(path, audio.T, SR, subtype="PCM_16")
    return path


def _small_chunks(settings):
    settings.hardware.chunk_seconds = 6.0
    settings.hardware.chunk_overlap_seconds = 1.0
    return settings


@pytest.mark.parametrize("total,chunk,overlap", [(10, 100, 5), (100, 30, 5), (1000, 300, 10), (301, 300, 10)])
def test_chunks_cover_everything_with_overlap(total, chunk, overlap):
    starts = chunk_starts(total, chunk, overlap)
    assert starts[0] == 0 and starts[-1] + chunk >= total
    for a, b in zip(starts, starts[1:]):
        assert b - a == chunk - overlap
    assert total - starts[-1] > overlap or len(starts) == 1


def test_prepare_source_declips_and_stores(tmp_path, music):
    project = Project.create(_recording(tmp_path, music), tmp_path / "projects")
    prepare_source(project)
    assert project.duration == pytest.approx(20.0, abs=0.01)
    assert project.state["analysis"]["declipped_samples"] > 0
    source = read_stem(project.stem_path("source"))
    assert np.abs(source).max() > 1.0  # rebuilt peaks above the old clip level survive storage


def test_separation_graph_and_seamless_chunks(tmp_path, music, fake_backend):
    settings = _small_chunks(Settings())
    project = Project.create(_recording(tmp_path, music, clip=False), tmp_path / "projects", settings)
    prepare_source(project)
    run_separation(project, fake_backend)
    project.reload()
    source = read_stem(project.stem_path("source"))

    music_part = source * 0.9
    instrumental = music_part * 0.6
    expected = {
        "crowd": source * 0.1,
        "drums": instrumental * 0.3,
        "other": instrumental * (0.1 + 0.05) * 0.8,  # other + vocal residue, minus woodwinds
        "woodwinds": instrumental * 0.15 * 0.2,
        "vocals": music_part * 0.4 * 0.8 * 0.9,
        "lead_vocals": music_part * 0.4 * 0.8 * 0.9 * 0.7,
        "kick": instrumental * 0.3 * 0.3,
    }
    for name, want in expected.items():
        got = read_stem(project.stem_path(name))
        assert got.shape == want.shape, name
        # Crossfading identical chunk results must be seamless: no error beyond 24-bit storage noise.
        assert np.max(np.abs(got - want)) < 1e-5, name


def test_resume_skips_finished_work_and_reruns_changed_passes(tmp_path, music, fake_backend):
    settings = _small_chunks(Settings())
    project = Project.create(_recording(tmp_path, music, clip=False), tmp_path / "projects", settings)
    prepare_source(project)
    run_separation(project, fake_backend)
    first = len(fake_backend.calls)

    run_separation(project, fake_backend)
    assert len(fake_backend.calls) == first  # everything cached

    settings.models.woodwinds_enabled = False
    settings.models.dereverb_model = "UVR-De-Echo-Normal.pth"
    project.set_settings(settings)
    run_separation(project, fake_backend)
    rerun = fake_backend.calls[first:]
    assert set(rerun) == {"UVR-De-Echo-Normal.pth", settings.models.denoise_model, settings.models.lead_backing_model}
    project.reload()
    assert project.state["aliases"]["other"] == "other_all"


def test_cancel_then_resume_from_the_same_chunk(tmp_path, music, fake_backend):
    settings = _small_chunks(Settings())
    project = Project.create(_recording(tmp_path, music, clip=False), tmp_path / "projects", settings)
    prepare_source(project)
    calls = {"n": 0}

    def cancel_after_three():
        calls["n"] += 1
        return calls["n"] > 3

    with pytest.raises(Cancelled):
        run_separation(project, fake_backend, cancel=cancel_after_three)
    done_before = len(fake_backend.calls)
    run_separation(project, fake_backend)
    crowd_calls = [c for c in fake_backend.calls if "crowd" in c]
    n_chunks = len(chunk_starts(project.frames, 6 * SR, SR))
    assert len(crowd_calls) == n_chunks  # no crowd chunk was processed twice
    assert done_before > 0
