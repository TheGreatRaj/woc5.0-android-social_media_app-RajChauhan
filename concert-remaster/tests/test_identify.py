import numpy as np
import pytest

from concert_remaster.identify import (
    Reference, ReferenceLibrary, _song_key, _split_artist_title, lyric_query, match_score, profile_similarity,
)

rng = np.random.default_rng(21)


def song(frames=240):
    """A chroma 'song': a chord progression held for a few frames each, like real harmony."""
    chords = rng.random((12, frames // 8 + 1)) ** 4
    chroma = np.repeat(chords, 8, axis=1)[:, :frames]
    return chroma / np.linalg.norm(chroma, axis=0, keepdims=True)


def live_version(chroma, stretch=1.1, shift=2, noise=0.15):
    frames = chroma.shape[1]
    idx = np.clip((np.arange(int(frames * stretch)) / stretch).astype(int), 0, frames - 1)
    warped = np.roll(chroma[:, idx], shift, axis=0) + rng.random((12, idx.size)) * noise
    return warped / np.linalg.norm(warped, axis=0, keepdims=True)


def test_match_score_is_key_and_tempo_invariant():
    original = song()
    score, shift = match_score(live_version(original, stretch=1.15, shift=3), original)
    unrelated, _ = match_score(live_version(song(), shift=3), original)
    assert score > 0.5 and shift == 3  # the reference rolled up 3 semitones matches the live key
    assert unrelated < 0.35


def test_a_preview_is_found_inside_the_whole_song():
    original = song(480)
    preview = original[:, 200:320]
    score, _ = match_score(live_version(original, stretch=0.95, shift=0), preview)
    assert score > 0.5


def _library(tmp_path, songs):
    library = ReferenceLibrary(tmp_path / "refs")
    for i, chroma in enumerate(songs):
        path = tmp_path / f"song{i}.mp3"
        path.write_bytes(b"x")
        library.add(Reference(f"k{i}", f"Title {i}", "Artist", str(path), "library"), signature=chroma)
    return library


def test_library_picks_the_right_song_and_rejects_unknown_ones(tmp_path):
    songs = [song() for _ in range(30)]
    library = _library(tmp_path, songs)
    ref, score = library.best_match(live_version(songs[17], stretch=1.08, shift=-1), min_score=0.45)
    assert ref is not None and ref.title == "Title 17"
    ref, score = library.best_match(live_version(song()), min_score=0.45)
    assert ref is None  # an unreleased song must not be forced onto a catalog title


def test_profile_prefilter_prefers_the_same_harmony():
    a = song()
    assert profile_similarity(live_version(a, shift=5), a) > profile_similarity(song(), a)


@pytest.mark.parametrize("stem,expected", [
    ("03 - Arijit Singh - Tum Hi Ho", ("Arijit Singh", "Tum Hi Ho")),
    ("Levels", ("", "Levels")),
])
def test_file_names_to_artist_and_title(stem, expected):
    assert _split_artist_title(stem) == expected


def test_song_keys_group_versions():
    assert _song_key('Tum Hi Ho (From "Aashiqui 2")') == _song_key("Tum Hi Ho") == "tum hi ho"


def test_lyric_query_picks_a_distinctive_line():
    text = "Oh oh oh. Hum tere bin ab reh nahi sakte, tere bina kya wajood mera! La la."
    assert lyric_query(text).startswith("Hum tere bin")
