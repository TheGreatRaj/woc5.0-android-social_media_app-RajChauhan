"""Damaged model downloads are found and fetched again instead of failing forever."""
import sys
import types

import pytest

torch = pytest.importorskip("torch")

from concert_remaster.separation import AudioSeparatorBackend, SeparationError, model_file_ok  # noqa: E402


def _checkpoint(path, legacy=False):
    weights = {"layer": torch.arange(50_000, dtype=torch.float32)}
    torch.save(weights, path, _use_new_zipfile_serialization=not legacy)
    return path


def _cut(path, fraction=0.6):
    data = path.read_bytes()
    path.write_bytes(data[: int(len(data) * fraction)])


@pytest.mark.parametrize("legacy", [False, True])
def test_complete_and_cut_off_checkpoints(tmp_path, legacy):
    path = _checkpoint(tmp_path / ("model.pth" if legacy else "model.ckpt"), legacy)
    assert model_file_ok(path)
    _cut(path)
    assert not model_file_ok(path)


def test_configs_are_not_parsed(tmp_path):
    config = tmp_path / "model_config.yaml"
    config.write_text("model: !!python/tuple [1, 2]\n")  # custom tags, like real model configs
    assert model_file_ok(config)
    (tmp_path / "empty.yaml").write_text("")
    assert not model_file_ok(tmp_path / "empty.yaml")


class FakeSeparator:
    """Loads like audio-separator: downloads a missing file, fails on a damaged one."""

    def __init__(self, model_dir, fail_first=0):
        self.model_dir, self.fail_first, self.downloads, self.loads = model_dir, fail_first, 0, 0

    def load_model(self, model_filename):
        self.loads += 1
        path = self.model_dir / model_filename
        if not path.exists():
            _checkpoint(path)
            self.downloads += 1
        if self.fail_first > 0 or not model_file_ok(path):
            self.fail_first -= 1
            raise RuntimeError("PytorchStreamReader failed reading zip archive: failed finding central directory")


def test_the_app_replaces_a_damaged_model_before_loading(tmp_path):
    backend = AudioSeparatorBackend(tmp_path / "models", tmp_path / "work", device="cpu")
    backend.model_dir.mkdir(parents=True)
    _cut(_checkpoint(backend.model_dir / "crowd.ckpt"))
    fake = FakeSeparator(backend.model_dir)
    backend._load(fake, ["crowd.ckpt"])
    assert fake.downloads == 1 and model_file_ok(backend.model_dir / "crowd.ckpt")


def test_damage_the_check_misses_is_repaired_once_then_explained(tmp_path):
    backend = AudioSeparatorBackend(tmp_path / "models", tmp_path / "work", device="cpu")
    backend.model_dir.mkdir(parents=True)
    _checkpoint(backend.model_dir / "vocals.ckpt")
    fake = FakeSeparator(backend.model_dir, fail_first=1)
    backend._load(fake, ["vocals.ckpt"])  # fails once, is fetched again, loads
    assert fake.loads == 2 and fake.downloads == 1
    fake = FakeSeparator(backend.model_dir, fail_first=5)
    with pytest.raises(SeparationError, match="run setup.bat again"):
        backend._load(fake, ["vocals.ckpt"])


def test_download_models_retries_an_incomplete_download(tmp_path, monkeypatch, capsys):
    from concert_remaster import cli

    attempts = {}

    class Separator:
        def __init__(self, model_file_dir, **kwargs):
            self.dir = tmp_path / "models"

        def download_model_and_data(self, model):
            attempts[model] = attempts.get(model, 0) + 1
            path = _checkpoint(self.dir / model)
            if attempts[model] == 1:
                _cut(path)  # the connection dropped the first time

    module = types.ModuleType("audio_separator.separator")
    module.Separator = Separator
    monkeypatch.setitem(sys.modules, "audio_separator.separator", module)
    monkeypatch.setattr(cli, "models_dir", lambda: tmp_path / "models")
    monkeypatch.setattr(cli.settings_mod, "PRESETS", {"tiny": {"models": {}}})
    monkeypatch.setattr("concert_remaster.separation.all_models", lambda models: ["crowd.ckpt"])
    monkeypatch.setattr(cli.settings_mod, "apply_preset", lambda s, name: s)
    (tmp_path / "models").mkdir()
    _cut(_checkpoint(tmp_path / "models" / "crowd.ckpt"))  # left over from an interrupted setup

    assert cli.download_models("tiny", whisper="") == 0
    out = capsys.readouterr().out
    assert "earlier download is incomplete" in out and "attempt 1 failed" in out
    assert attempts["crowd.ckpt"] == 2 and model_file_ok(tmp_path / "models" / "crowd.ckpt")
