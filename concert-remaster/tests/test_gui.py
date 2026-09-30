import io
from urllib.parse import quote

import numpy as np
import pytest
import soundfile as sf
from conftest import SR

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from concert_remaster.engine import Project  # noqa: E402
from concert_remaster.settings import Settings  # noqa: E402
from concert_remaster.workflow import Job, renumber  # noqa: E402


@pytest.fixture
def client(tmp_path, monkeypatch, music, fake_backend):
    monkeypatch.setenv("CONCERT_REMASTER_HOME", str(tmp_path))
    import importlib

    import concert_remaster.gui.server as server

    importlib.reload(server)
    source = tmp_path / "show.wav"
    sf.write(source, np.tile(music, 3)[:, : 16 * SR].T, SR, subtype="PCM_16")
    settings = Settings()
    settings.identify.enabled = False
    settings.speech.transcribe = False
    project = Project.create(source, tmp_path / "projects", settings, name="Test Show")
    Job(project, backend=fake_backend).analyze()
    project.update(lambda s: s.__setitem__("segments", renumber([
        {"kind": "song", "start": 0.0, "end": 10.0}, {"kind": "crowd", "start": 10.0, "end": 16.0}])))
    with TestClient(server.create_app()) as c:
        c.project_id = project.root.name
        c.source = source
        yield c


def url(c, suffix=""):
    return f"/api/projects/{quote(c.project_id)}{suffix}"


def test_index_and_schema(client):
    assert "Concert Remaster" in client.get("/").text
    schema = client.get("/api/schema").json()
    assert {g["name"] for g in schema["groups"]} >= {"hardware", "models", "speech", "master", "output"}
    assert "ultra" in schema["presets"]


def test_project_listing_and_detail(client):
    projects = client.get("/api/projects").json()
    assert projects[0]["name"] == "Test Show" and projects[0]["songs"] == 1
    detail = client.get(url(client)).json()
    assert detail["segments"][0]["title"] == "Song 01"
    assert "drums" in detail["aliases"]


def test_segment_edits_are_renumbered_and_validated(client):
    segs = client.get(url(client)).json()["segments"]
    segs[1]["kind"] = "song"
    segs[1]["title"] = "Encore"
    saved = client.put(url(client, "/segments"), json=segs).json()
    assert [s["track"] for s in saved] == [1, 2] and saved[1]["title"] == "Encore"
    segs[0]["kind"] = "banana"
    assert client.put(url(client, "/segments"), json=segs).status_code == 400


def test_waveform_audio_and_preview(client):
    peaks = client.get(url(client, "/peaks")).json()
    assert peaks["rate"] == 10 and len(peaks["peaks"]) == pytest.approx(160, abs=2)
    wav = client.get(url(client, "/audio"), params={"stem": "drums+bass", "start": 1, "end": 3})
    data, rate = sf.read(io.BytesIO(wav.content))
    assert rate == SR and data.shape[0] == 2 * SR
    seg = client.get(url(client)).json()["segments"][0]["id"]
    preview = client.post(url(client, "/preview"), json={"segment": seg, "start": 0, "seconds": 5})
    data, _ = sf.read(io.BytesIO(preview.content))
    assert data.shape[0] == 5 * SR and np.abs(data).max() > 0.01


def test_settings_roundtrip_and_presets(client):
    settings = client.get(url(client)).json()["settings"]
    settings["master"]["target_lufs"] = -9.5
    assert client.put(url(client, "/settings"), json=settings).json()["ok"]
    assert client.get(url(client)).json()["settings"]["master"]["target_lufs"] == -9.5
    fast = client.post("/api/preset", json={"settings": settings, "preset": "fast"}).json()
    assert fast["models"]["vocal_model"] == "" and fast["master"]["target_lufs"] == -9.5


def test_files_cannot_escape_the_output_folder(client):
    assert client.get(f"/files/{quote(client.project_id)}/..%2Fproject.json").status_code == 404
    assert client.get("/api/projects/..%2F..%2Fetc").status_code == 404


def test_new_project_from_missing_file_is_rejected(client):
    assert client.post("/api/projects", json={"source": "/nope/missing.mp4", "start": False}).status_code == 400
