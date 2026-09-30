import json

import numpy as np
import pytest
import soundfile as sf
from conftest import SR

from concert_remaster.cli import main
from concert_remaster.pipeline import RemasterSettings, remaster
from concert_remaster.separation import (
    MDX_CROWD,
    AudioSeparatorBackend,
    SeparationError,
    build_plan,
    normalize_stem_name,
    separate_concert,
)


class TestPlans:
    def test_default_six_stem_plan(self):
        plan = build_plan("balanced", 6)
        assert plan.instruments == "htdemucs_6s.yaml"
        assert plan.vocals and plan.crowd and plan.dereverb

    def test_fast_uses_one_demucs_pass_for_vocals(self):
        plan = build_plan("fast", 4)
        assert plan.vocals is None and plan.instruments == "htdemucs.yaml"

    def test_fast_two_stem_still_needs_a_vocal_model(self):
        plan = build_plan("fast", 2)
        assert plan.vocals and plan.instruments is None

    def test_overrides(self):
        plan = build_plan("best", 6, crowd="none", dereverb="my.ckpt")
        assert plan.crowd is None and plan.dereverb == "my.ckpt"

    @pytest.mark.parametrize("kwargs", [{"preset": "nope"}, {"stems": 3}, {"bogus": "x"}])
    def test_invalid(self, kwargs):
        with pytest.raises(ValueError):
            build_plan(**kwargs)

    def test_cannot_disable_all_separation(self):
        with pytest.raises(ValueError):
            build_plan("fast", 4, instruments="none")


@pytest.mark.parametrize("raw,expected", [("No Crowd", "nocrowd"), ("Vocals", "vocals"), ("noreverb", "noreverb"), ("No Reverb", "noreverb")])
def test_stem_names(raw, expected):
    assert normalize_stem_name(raw) == expected


class TestSeparateConcert:
    def test_six_stem_flow(self, music, fake_backend):
        plan = build_plan("best", 6)
        result = separate_concert(music, SR, plan, fake_backend)
        assert set(result.stems) == {"crowd", "vocals", "drums", "bass", "guitar", "piano", "other"}
        assert fake_backend.calls == [plan.crowd, plan.vocals, plan.instruments, plan.dereverb, plan.denoise]
        # Everything except removed reverb and noise adds back up to the recording.
        total = sum(result.stems.values()) + sum(result.removed.values())
        np.testing.assert_allclose(total, music, atol=1e-5)

    def test_leftover_vocal_goes_to_other_not_vocal_stem(self, music, fake_backend):
        plan = build_plan("balanced", 6, crowd="none", dereverb="none")
        result = separate_concert(music, SR, plan, fake_backend)
        instrumental = music * 0.6
        np.testing.assert_allclose(result.stems["vocals"], music * 0.4, atol=1e-6)
        np.testing.assert_allclose(result.stems["other"], instrumental * 0.15, atol=1e-6)

    def test_fast_takes_vocals_from_demucs(self, music, fake_backend):
        result = separate_concert(music, SR, build_plan("fast", 4, crowd="none"), fake_backend)
        np.testing.assert_allclose(result.stems["vocals"], music * 0.25, atol=1e-6)
        assert "instrumental" not in result.stems

    def test_two_stems(self, music, fake_backend):
        result = separate_concert(music, SR, build_plan("balanced", 2), fake_backend)
        assert set(result.stems) == {"crowd", "vocals", "instrumental"}

    def test_unexpected_model_output(self, music):
        class Weird:
            def separate(self, audio, sample_rate, model):
                return {"bananas": audio}

        with pytest.raises(SeparationError, match="bananas"):
            separate_concert(music, SR, build_plan("balanced", 2), Weird())


class FakeSeparator:
    """Mimics audio-separator's file interface: reads the input, writes named stems."""

    def __init__(self, output_dir):
        self.output_dir = output_dir

    def load_model(self, model_filename):
        self.model = model_filename

    def separate(self, path):
        audio, sr = sf.read(path, dtype="float32")
        base = path.rsplit("/", 1)[-1][:-4]
        outputs = []
        for stem, share in (("No Crowd", 0.7), ("Crowd", 0.3)):
            name = f"{base}_({stem})_{self.model[:-5]}.wav"
            sf.write(f"{self.output_dir}/{name}", audio * share, sr, subtype="FLOAT")
            outputs.append(name)
        return outputs


def test_audio_separator_backend_keeps_levels_exact(tmp_path, music):
    backend = AudioSeparatorBackend(tmp_path / "models", tmp_path)
    backend._separator = FakeSeparator(str(tmp_path))
    loud = music * 3.0  # peaks above 1.0 must survive the trip through files
    stems = backend.separate(loud, SR, MDX_CROWD)
    assert set(stems) == {"nocrowd", "crowd"}
    np.testing.assert_allclose(stems["crowd"], loud * 0.3, rtol=1e-5, atol=1e-6)
    assert not list(tmp_path.glob("*.wav")), "work files should be cleaned up"


def _write_recording(path, music):
    # Clip it like an overloaded phone mic.
    sf.write(path, np.clip(music * 3.0, -1, 1).T, SR, subtype="PCM_16")


def test_remaster_end_to_end(tmp_path, music, fake_backend):
    source = tmp_path / "gig.wav"
    _write_recording(source, music)
    settings = RemasterSettings(preset="best", stems=6, export_raw_stems=True, target_lufs=-12.0)
    result = remaster(source, tmp_path / "out", settings, backend=fake_backend)

    assert result.master_path.name == "gig (Remastered).wav"
    mastered, sr = sf.read(result.master_path, always_2d=True)
    assert sr == SR and mastered.shape == (music.shape[1], 2)
    report = json.loads(result.report_path.read_text())
    assert report["master"]["output_lufs"] == pytest.approx(-12.0, abs=0.5)
    assert report["master"]["true_peak_dbtp"] <= -1.0
    assert report["declipped_samples"] > 0
    assert set(report["stems"]) == {"vocals", "drums", "bass", "guitar", "piano", "other"}
    assert {p.stem for name, p in result.stem_paths.items() if not name.startswith("raw/")} == set(report["stems"])
    assert (tmp_path / "out" / "gig" / "stems_raw" / "crowd.wav").exists()
    assert (tmp_path / "out" / "gig" / "stems_raw" / "vocal_reverb.wav").exists()


def test_keep_crowd_blends_audience_back(tmp_path, music, fake_backend):
    source = tmp_path / "gig.wav"
    _write_recording(source, music)
    settings = RemasterSettings(preset="balanced", stems=2, crowd_db=-15.0, export_stems=False)
    result = remaster(source, tmp_path, settings, backend=fake_backend)
    assert result.stem_paths == {}
    stems = result.report["stems"]
    vocal = stems["vocals"]["loudness_lufs"] + stems["vocals"]["gain_db"]
    crowd = stems["crowd"]["loudness_lufs"] + stems["crowd"]["gain_db"]
    assert crowd - vocal == pytest.approx(-15.0, abs=0.05)


def test_cli_lists_presets(capsys):
    assert main(["--list-presets"]) == 0
    out = capsys.readouterr().out
    assert "balanced" in out and "htdemucs_6s.yaml" in out


def test_cli_reports_failures(tmp_path, capsys):
    assert main([str(tmp_path / "missing.wav"), "-o", str(tmp_path)]) == 1
    assert "FAILED" in capsys.readouterr().err


def test_cli_rejects_bad_gain():
    with pytest.raises(SystemExit):
        main(["x.wav", "--gain", "vocals"])

