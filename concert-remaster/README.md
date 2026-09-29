# concert-remaster 🎤→🎧

Turn a phone or handheld recording of a live show into something that sounds like the studio release:
crowd noise gone, venue echo stripped off the vocal, every instrument split out and cleaned up, then
remixed and mastered to streaming loudness.

Everything runs **on your own machine**. The AI models (Roformer, MDX-Net, VR and Demucs) are
downloaded once and then work offline on CPU, NVIDIA GPUs (CUDA) or Apple Silicon (MPS).

```
recording.mp4
   │
   ├─ decode, rebuild clipped peaks, remove rumble
   ├─ AI: remove crowd noise ─────────────────────────────► crowd (optional blend-back)
   ├─ AI: isolate the vocal
   │     └─ AI: strip venue reverb → AI: denoise
   ├─ AI: split the band → drums · bass · guitar · piano · other
   ├─ studio chain per stem (EQ, gate, compression, de-ess, stereo placement, plate reverb)
   ├─ auto-mix toward a studio balance
   └─ master: glue compression, tonal-balance EQ, true-peak limiter, -14 LUFS
         │
         ▼
   "Song (Remastered).wav"  +  stems/  +  report.json
```

## Install

You need Python 3.10+ and [ffmpeg](https://ffmpeg.org/download.html) (`brew install ffmpeg`,
`sudo apt install ffmpeg`, or `winget install ffmpeg`).

```bash
cd concert-remaster
pip install -e ".[cpu]"      # any computer, including Apple Silicon (uses the GPU through MPS)
# or
pip install -e ".[gpu]"      # NVIDIA GPU with CUDA
```

The first run downloads the models it needs (roughly 0.5–1 GB per model) into
`~/.cache/concert-remaster/models`. After that no internet connection is needed.

## Use it

```bash
concert-remaster "Live at the Roundhouse.mp4"
```

Results land in `remastered/<name>/`:

| File | What it is |
| --- | --- |
| `<name> (Remastered).wav` | The finished master, 24-bit, -14 LUFS, -1 dBTP peak |
| `stems/vocals.wav`, `drums.wav`, `bass.wav`, `guitar.wav`, `piano.wav`, `other.wav` | Cleaned, processed stems, balanced exactly as in the mix, so they line up in any DAW |
| `stems_raw/` | (with `--raw-stems`) unprocessed model outputs, the crowd, and the reverb that was removed |
| `report.json` | Models used, loudness of each stem, gains applied, final loudness and peak |

More examples:

```bash
# Several songs at once, FLAC output
concert-remaster song1.m4a song2.m4a song3.m4a -o remastered --format flac

# Make it sound like the studio version: match the tonal balance of a reference track
concert-remaster live.wav --reference "studio version.flac"

# Keep a bit of the audience for a live-album feel (15 LU under the vocal)
concert-remaster live.wav --keep-crowd -15

# Louder master, vocal up 2 dB, drums down 1 dB
concert-remaster live.wav --target-lufs -10 --gain vocals=2 --gain drums=-1

# Best quality (use a GPU), or a fast preview
concert-remaster live.wav --preset best
concert-remaster live.wav --preset fast --stems 4

# See every option / every model
concert-remaster --help
concert-remaster --list-presets
```

It also works as a library:

```python
from concert_remaster import RemasterSettings, remaster

result = remaster("gig.mp4", "remastered", RemasterSettings(preset="best", crowd_db=-18))
print(result.master_path, result.report["master"]["output_lufs"])
```

## Presets and speed

| Preset | Crowd removal | Vocal isolation | Vocal clean-up | Band split |
| --- | --- | --- | --- | --- |
| `fast` | MDX-Net Crowd HQ | Demucs (one pass) | none | Demucs |
| `balanced` (default) | MDX-Net Crowd HQ | Mel-Band Roformer (Kim) | VR DeEcho-DeReverb | Demucs 6-stem / fine-tuned 4-stem |
| `best` | Mel-Band Roformer Crowd | BS-Roformer (ViperX 1297) | Mel-Roformer de-reverb + denoise | Demucs, 2 shifts |

`--stems 6` (default) gives vocals, drums, bass, guitar, piano and other. `--stems 4` merges guitar and
piano into "other" using the higher-quality fine-tuned Demucs. `--stems 2` is just vocals + band.

Roformer models are heavy. Measured on a 4-core cloud VM with no GPU, each second of audio took
about 1.3 s for MDX-Net crowd removal, 0.8 s for VR de-reverb, 6 s for the Mel-Roformer crowd model
and 18 s for BS-Roformer. A full song on `best` wants a GPU; `balanced` is the practical CPU choice,
and `--preset fast` is for quick previews. Roformers process audio in overlapping 8-second windows:
`balanced` uses 2 passes per window, and on a GPU you can raise `--overlap` to 4–8 for a slightly
cleaner result. For very long recordings, `--chunk-seconds 300` keeps memory in check.

## What each step does

**Restoration.** Loud gigs overload phone microphones. Flat-topped, clipped peaks are detected and
redrawn with a cubic spline before anything else touches the waveform, then DC offset and
handling/wind rumble under 25 Hz are removed.

**AI separation.** Crowd noise is taken out of the whole recording first, since it smears across
every stem otherwise. The vocal is isolated next, and only the vocal gets de-reverb and denoise
models, because those are trained on voice and damage instruments. The instrumental is then split
into instruments. Any vocal leftovers the instrument model finds go into "other" so the vocal stem
stays clean. Models are fed float audio with headroom and their outputs are scaled back, so stem
levels stay sample-accurate and sum back to the recording.

**Studio chain per stem.** Each instrument gets what an engineer would put on its track. A
downward expander pushes bleed and room wash under the music. Corrective EQ cuts the boxy
250–400 Hz build-up that rooms add, and tone EQ adds presence and air. Compression with thresholds
relative to the stem's own level means it behaves the same on quiet or hot separations. The vocal
is de-essed (split band, only above 5.5 kHz). Kick and bass are centred below 110–150 Hz, and
mono-recorded guitars and keys are widened with a mono-compatible decorrelator. The vocal gets a
short plate reverb to replace the venue reverb that was removed.

**Auto-mix.** Loudness (LUFS) of each stem is measured and moved partway toward a studio balance
(drums 2 LU under the vocal, bass and guitar 4 LU, keys 5 LU), at most 6 dB per stem. Stems where the
model found nothing (a piano stem on a song with no piano) are detected and turned down, so their
separation noise doesn't get boosted.

**Mastering.** 2:1 bus compression glues the stems back together. A linear-phase EQ pulls the
overall tilt toward the slope typical of commercial releases and cuts narrow resonances (room
modes). With `--reference` it matches the reference's spectrum instead. A look-ahead limiter
working on 4× oversampled true peaks hits the loudness target without exceeding -1 dBTP. If
reaching the target would take more than 6 dB of limiting, it stops short rather than crushing
the song.

## Honest limitations

- AI separation is very good but not perfect. Busy, distorted or very reverberant recordings leave
  artifacts, especially in the guitar/piano/other stems.
- It can't restore what the microphone never captured. Phone mics roll off deep bass, and heavy
  analog distortion (a mic capsule overloaded before the digital stage) can't be undone. Low end is
  rebalanced, not invented.
- Audience singing along is musically similar to the lead vocal and can survive crowd removal.
- Process one song per file for best results. Very long files (full sets) need a lot of RAM.
- Only remaster recordings you have the rights to share. Concert recordings usually contain other
  people's copyrighted performances.

## Models and licences

Separation runs through [python-audio-separator](https://github.com/nomadkaraoke/python-audio-separator)
(MIT). The model weights come from the [Ultimate Vocal Remover](https://github.com/Anjok07/ultimatevocalremovergui)
community and [Demucs](https://github.com/facebookresearch/demucs) (MIT). Check each model's licence
before commercial use.

## Development

```bash
pip install -e ".[dev]"
pytest
```

The test suite uses a fake separation backend, so it runs in seconds without downloading any models.
