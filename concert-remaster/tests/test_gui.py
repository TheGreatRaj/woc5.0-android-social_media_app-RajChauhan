import time
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
        c.projects_root = tmp_path / "projects"
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


def test_waveforms_and_mixer_tracks(client):
    peaks = client.get(url(client, "/peaks")).json()
    assert peaks["rate"] == 10 and len(peaks["peaks"]) == pytest.approx(160, abs=2)
    drums = client.get(url(client, "/peaks"), params={"stem": "drums", "start": 2, "end": 6, "rate": 20}).json()
    assert drums["rate"] == 20 and len(drums["peaks"]) == 80 and max(drums["peaks"]) > 0.01
    seg = client.get(url(client)).json()["segments"][0]["id"]
    info = client.get(url(client, f"/tracks/{seg}")).json()
    assert info["studio"] == "missing" and "drums" in info["raw_tracks"] and "crowd" in info["raw_tracks"]
    assert info["tracks"]["crowd"]["mute"] and not info["tracks"]["drums"]["mute"]


def _wait(predicate, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


def test_native_player_plays_mixes_and_previews(client, monkeypatch):
    monkeypatch.setenv("CONCERT_REMASTER_AUDIO", "null")
    seg = client.get(url(client)).json()["segments"][0]["id"]
    status = client.post("/api/player", json={"cmd": "song", "project": client.project_id, "seg": seg, "source": "raw",
                                               "gains": {"drums": 1.0, "bass": 0.5}, "position": 2.0, "play": True}).json()
    assert status["kind"] == "song" and status["playing"] and status["start"] == 0.0
    assert _wait(lambda: client.get("/api/player").json()["position"] > 2.2)
    levels = client.get("/api/player").json()["levels"]
    assert levels["drums"] > -40 and levels["bass"] > -60
    client.post("/api/player", json={"cmd": "mix", "gains": {"drums": 0.0}})
    assert _wait(lambda: client.get("/api/player").json()["levels"].get("drums", 0) < -100)
    paused = client.post("/api/player", json={"cmd": "pause"}).json()
    assert not paused["playing"]
    clip = client.post("/api/player", json={"cmd": "clip", "project": client.project_id, "stems": ["source"], "start": 1, "end": 3}).json()
    assert clip["kind"] == "clip" and clip["playing"] and clip["end"] == 3
    preview = client.post(url(client, "/preview"), json={"segment": seg, "start": 0, "seconds": 5}).json()
    assert preview["kind"] == "preview" and preview["end"] - preview["start"] == pytest.approx(5, abs=0.01)
    client.post("/api/player", json={"cmd": "stop"})


def test_studio_tracks_are_prepared_and_played(client, fake_backend, monkeypatch):
    monkeypatch.setenv("CONCERT_REMASTER_AUDIO", "null")
    project = Project(client.projects_root / client.project_id)
    seg = project.state["segments"][0]
    assert Job(project, backend=fake_backend).prepare_studio() == [seg["id"]]
    info = client.get(url(client, f"/tracks/{seg['id']}")).json()
    assert info["studio"] == "ready" and "crowd" in info["studio_tracks"]
    gains = {n: t["auto_gain_db"] for n, t in info["tracks"].items()}
    assert gains["lead_vocals"] == 0.0 and any(abs(g) > 0.1 for g in gains.values())
    status = client.post("/api/player", json={"cmd": "song", "project": client.project_id, "seg": seg["id"],
                                               "source": "studio", "play": True}).json()
    assert status["info"]["source"] == "studio" and set(status["levels"]) <= set(info["studio_tracks"])
    # Changing a setting that shapes the tracks makes them out of date.
    settings = client.get(url(client)).json()["settings"]
    settings["stems"]["drums"]["compress_ratio"] = 6.0
    client.put(url(client, "/settings"), json=settings)
    assert client.get(url(client, f"/tracks/{seg['id']}")).json()["studio"] == "stale"
    client.post("/api/player", json={"cmd": "stop"})


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


def test_older_projects_get_new_settings_with_defaults(client):
    project = Project(client.projects_root / client.project_id)
    project.update(lambda s: s["settings"].pop("effects"))
    settings = client.get(url(client)).json()["settings"]
    assert settings["effects"]["action"] == "remove" and settings["effects"]["co2"] is True
