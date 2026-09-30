import numpy as np
import pytest
from conftest import SR, stereo, tone

from concert_remaster.analysis import active_level_db, integrated_lufs, is_mono, true_peak_db
from concert_remaster.audio_io import load_audio, save_audio, to_stereo
from concert_remaster.restoration import declip, remove_rumble, spectral_denoise
from concert_remaster.stem_processing import PROFILES, deess, mono_below, process_stem, pseudo_stereo, set_width


def rms(x):
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64))))


class TestAudioIO:
    @pytest.mark.parametrize("suffix,bit_depth", [(".wav", 24), (".flac", 16), (".wav", "float")])
    def test_roundtrip(self, tmp_path, suffix, bit_depth):
        audio = stereo(tone(440))
        path = save_audio(tmp_path / f"a{suffix}", audio, SR, bit_depth)
        loaded = load_audio(path)
        assert loaded.shape == audio.shape
        assert np.max(np.abs(loaded - audio)) < 1e-3

    def test_mp3_roundtrip(self, tmp_path):
        path = save_audio(tmp_path / "a.mp3", stereo(tone(440)), SR)
        loaded = load_audio(path)
        assert loaded.shape[0] == 2
        assert abs(loaded.shape[1] - SR * 2) < SR * 0.1

    def test_resamples_and_upmixes(self, tmp_path):
        import soundfile as sf

        sf.write(tmp_path / "mono48k.wav", tone(440, sr=48000), 48000)
        loaded = load_audio(tmp_path / "mono48k.wav")
        assert loaded.shape[0] == 2
        assert abs(loaded.shape[1] - SR * 2) <= 2

    def test_to_stereo_drops_extra_channels(self):
        assert to_stereo(np.zeros((6, 10))).shape == (2, 10)

    def test_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_audio(tmp_path / "nope.wav")


class TestRestoration:
    def test_rumble_removed_music_kept(self):
        audio = stereo(tone(8, amp=0.5) + tone(440, amp=0.2) + 0.1)
        out = remove_rumble(audio, SR)
        assert abs(out.mean()) < 1e-3
        assert rms(out[:, SR:]) == pytest.approx(0.2 / np.sqrt(2), rel=0.05)

    def test_declip_restores_peaks(self):
        clean = stereo(tone(200, amp=1.4))
        clipped = np.clip(clean, -1.0, 1.0)
        repaired, count = declip(clipped)
        assert count > 0
        assert np.abs(repaired).max() > 1.2
        assert rms(repaired - clean) < 0.3 * rms(clipped - clean)

    def test_declip_leaves_clean_audio_alone(self, music):
        faded = stereo(tone(200, amp=0.5) * np.linspace(0.2, 1.0, SR * 2, dtype=np.float32))
        for audio in (faded, music):
            repaired, count = declip(audio)
            assert count == 0
            assert np.array_equal(repaired, audio)

    def test_declip_steady_sine_crests_are_harmless(self):
        # A constant-level sine has identical curved crests; even if some are
        # flagged, the spline must redraw the same curve.
        audio = stereo(tone(100, amp=0.5))
        repaired, _ = declip(audio)
        assert np.max(np.abs(repaired - audio)) < 0.01

    def test_denoise_reduces_hiss_between_phrases(self):
        rng = np.random.default_rng(0)
        phrase = tone(440, seconds=4.0, amp=0.3) * (np.arange(SR * 4) % SR < SR // 2)
        hiss = rng.standard_normal(SR * 4).astype(np.float32) * 0.01
        out = spectral_denoise(stereo(phrase + hiss), SR, strength=0.6)
        gaps = (np.arange(SR * 4) % SR) > SR * 0.6
        assert rms(out[:, gaps]) < 0.5 * rms(hiss[gaps])
        assert rms(out) == pytest.approx(rms(phrase), rel=0.1)


class TestStemProcessing:
    @pytest.mark.parametrize("name", sorted(PROFILES))
    def test_every_profile_runs(self, music, name):
        out = process_stem(music, SR, PROFILES[name])
        assert out.shape == music.shape and out.dtype == np.float32
        assert np.isfinite(out).all()
        assert rms(out) > 0.01

    def test_silent_stem_passes_through(self):
        silent = np.zeros((2, SR), dtype=np.float32)
        assert not process_stem(silent, SR, PROFILES["vocals"]).any()

    def test_mono_below_centres_bass_only(self):
        left = tone(50) + tone(2000)
        right = -tone(50) + 0.2 * tone(2000)
        out = mono_below(np.stack([left, right]), SR, 150.0)
        side = out[0] - out[1]
        # The 50 Hz part was fully out of phase; after mono-ing only the 2 kHz difference remains.
        assert rms(side[SR // 2 : -SR // 2]) == pytest.approx(rms(0.8 * tone(2000)), rel=0.05)

    def test_width(self):
        audio = np.stack([tone(300), 0.5 * tone(300)])
        assert is_mono(set_width(audio, 0.0))
        wide = set_width(audio, 2.0)
        assert rms(wide[0] - wide[1]) == pytest.approx(2 * rms(audio[0] - audio[1]), rel=1e-4)

    def test_pseudo_stereo_folds_down_to_mono(self):
        mono = stereo(tone(1000) + tone(3000, amp=0.1))
        out = pseudo_stereo(mono, SR)
        assert not is_mono(out)
        np.testing.assert_allclose(out.mean(axis=0), mono[0], atol=1e-6)

    def test_deess_tames_sibilance_only(self):
        from concert_remaster.restoration import highpass, lowpass

        body = tone(300, seconds=2.0, amp=0.1)
        rng = np.random.default_rng(3)
        ess = np.zeros_like(body)
        for start in range(SR // 4, 2 * SR, SR // 2):
            ess[start : start + 3000] = rng.standard_normal(3000) * 0.3
        ess = highpass(ess[None], SR, 6000.0, order=4)[0]
        voice = stereo(body + ess)
        out = deess(voice, SR)
        assert rms(highpass(out, SR, 6000.0, order=4)) < 0.6 * rms(ess)
        low_in, low_out = lowpass(voice, SR, 2000.0, order=4), lowpass(out, SR, 2000.0, order=4)
        assert rms(low_out - low_in) < 0.02 * rms(low_in)


class TestAnalysis:
    def test_lufs_of_full_scale_sine(self):
        # BS.1770 calibration: a 0 dBFS 1 kHz sine reads -3.01 LUFS per channel, so 0 LUFS on two.
        assert integrated_lufs(stereo(tone(997, seconds=3.0, amp=1.0)), SR) == pytest.approx(0.0, abs=0.2)

    def test_silence(self):
        silent = np.zeros((2, SR), dtype=np.float32)
        assert integrated_lufs(silent, SR) == -np.inf
        assert active_level_db(silent, SR) is None

    def test_true_peak_sees_intersample_overs(self):
        # Samples of a fs/4 sine at 45 degrees phase never hit the true peak.
        n = np.arange(SR)
        x = np.sin(np.pi / 2 * n + np.pi / 4).astype(np.float32)
        sample_peak = 20 * np.log10(np.abs(x).max())
        assert true_peak_db(stereo(x)) > sample_peak + 2.5
