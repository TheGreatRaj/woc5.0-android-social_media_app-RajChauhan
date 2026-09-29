import numpy as np
import pytest
from conftest import SR, stereo, tone

from concert_remaster.analysis import integrated_lufs, true_peak_db
from concert_remaster.mastering import apply_eq_curve, band_spectrum_db, limit, master, tonal_correction_db
from concert_remaster.mixing import auto_balance, sum_stems


def test_auto_balance_lifts_buried_vocal(music):
    stems = {"vocals": music * 0.1, "drums": music, "bass": music * 0.8}
    result = auto_balance(stems, SR, strength=1.0, max_adjust_db=30.0)
    after = {name: result.loudness_lufs[name] + result.gains_db[name] for name in stems}
    # Studio template: drums 2 LU under the vocal, bass 4 LU under.
    assert after["drums"] - after["vocals"] == pytest.approx(-2.0, abs=0.01)
    assert after["bass"] - after["vocals"] == pytest.approx(-4.0, abs=0.01)
    assert result.reference == "vocals"


def test_auto_balance_is_partial_and_capped(music):
    stems = {"vocals": music * 0.05, "drums": music}  # vocal 26 dB down: buried, not a ghost
    result = auto_balance(stems, SR, strength=0.5, max_adjust_db=6.0)
    assert result.gains_db["drums"] == pytest.approx(-6.0)
    assert result.gains_db["vocals"] == 0.0


def test_ghost_stems_are_turned_down(music):
    stems = {"vocals": music, "piano": music * 1e-3, "empty": np.zeros_like(music)}
    result = auto_balance(stems, SR, user_gains_db={"piano": 1.0})
    assert set(result.ghosts) == {"piano", "empty"}
    assert result.gains_db["piano"] == pytest.approx(-6.0 + 1.0)


def test_sum_stems_applies_gains(music):
    mix = sum_stems({"a": music, "b": music}, {"a": 0.0, "b": -120.0})
    np.testing.assert_allclose(mix, music, atol=1e-5)


def test_limiter_holds_true_peak_ceiling(music):
    loud = music * 6.0
    out, reduction = limit(loud, SR, ceiling_db=-1.0)
    assert reduction > 6.0
    assert true_peak_db(out) <= -0.95


def test_limiter_is_transparent_below_ceiling(music):
    quiet = music * 0.1
    out, reduction = limit(quiet, SR, ceiling_db=-1.0)
    assert reduction == 0.0
    assert np.array_equal(out, quiet)


@pytest.mark.parametrize("target", [-16.0, -12.0])
def test_master_hits_loudness_and_ceiling(music, target):
    out, report = master(music * 0.05, SR, target_lufs=target, ceiling_dbtp=-1.0)
    assert report.output_lufs == pytest.approx(target, abs=0.5)
    assert integrated_lufs(out, SR) == pytest.approx(report.output_lufs, abs=0.01)
    assert report.true_peak_dbtp <= -1.0


def test_master_refuses_to_crush(music):
    _, report = master(music, SR, target_lufs=0.0, max_limiting_db=4.0)
    assert report.max_limiting_db <= 4.5
    assert report.output_lufs < 0.0


def test_flat_eq_curve_is_identity(music):
    centers, _ = band_spectrum_db(music, SR)
    out = apply_eq_curve(music, SR, centers, np.zeros_like(centers))
    np.testing.assert_allclose(out, music, atol=2e-3)


def test_reference_match_moves_toward_reference():
    rng = np.random.default_rng(5)
    noise = rng.standard_normal((2, SR * 4)).astype(np.float32) * 0.1
    from concert_remaster.restoration import lowpass

    dark = lowpass(noise, SR, 2000.0)
    centers, curve = tonal_correction_db(noise, SR, reference=dark, strength=1.0, max_db=6.0)
    assert curve[centers > 6000].mean() < -4.0
    assert abs(curve[(centers > 200) & (centers < 1000)].mean()) < 1.5


def test_auto_tonal_balance_darkens_thin_bright_recording():
    rng = np.random.default_rng(6)
    white = rng.standard_normal((2, SR * 4)).astype(np.float32) * 0.05
    centers, curve = tonal_correction_db(white, SR)
    assert curve[centers > 8000].mean() < curve[(centers > 100) & (centers < 300)].mean()
    assert np.abs(curve).max() <= 4.0
