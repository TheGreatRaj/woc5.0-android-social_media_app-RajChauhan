# Concert Remaster 🎤→🎧

Turn your phone, watch or voice-recorder recordings of live shows into something that sounds like
the released album: crowd noise and venue echo removed, every instrument separated into clean stems,
each song found, named and matched to the tone of its studio version, then mixed and mastered.
Built for whole concerts: 2–3 hour MP3/MP4/M4A recordings, bands or DJ sets.

**All the processing runs on your own PC** with local AI models (NVIDIA via CUDA, AMD via DirectML,
or the CPU). The internet is only used, if you allow it, to look songs up and fetch their studio
versions, which are then kept locally.

## Install on Windows (one time)

1. Download this repository as a ZIP and extract it anywhere with ~20 GB free.
2. Double-click **`setup.bat`** (in the top folder). It:
   - detects your graphics card (NVIDIA → CUDA, AMD → DirectML, otherwise CPU),
   - installs a private Python 3.12 and the AI libraries for that card,
   - installs ffmpeg,
   - downloads **every AI model (~12 GB) and Whisper large-v3**, so the app then works fully offline,
   - puts a **Concert Remaster** shortcut on your desktop.

   Everything goes inside the `concert-remaster` folder; nothing is installed system-wide. If
   anything fails (e.g. the connection drops), run `setup.bat` again; it continues where it stopped.
   To choose the device yourself: `setup.bat -Gpu nvidia` / `-Gpu amd` / `-Gpu cpu`.
3. Start the app with the desktop shortcut or **`start.bat`**.

For your machines: the **laptop (RTX 3060)** is the one for full-quality runs of long shows. The
**RX 580 desktop** uses DirectML, which is experimental: models that don't run there fall back to
the 14400F CPU automatically, which is much slower.

## Using the app

1. **New project** → pick the recording (the file picker accepts MP3, MP4, M4A, WAV, FLAC and most
   video files), choose the quality preset, optionally the artist's name, the show type, and what to
   do when the artist talks. Analysis starts.
2. **Analyze** runs on its own; you can close the window or press Stop at any time. Every finished
   piece is kept, so pressing Continue later picks up where it stopped (even after a restart).
3. **Review** the timeline: songs (purple), the artist talking (amber), applause (green). Drag the
   edges, split or merge parts, fix titles, choose per part what happens (keep the talk with a clearer
   voice, cut it, or mute the voice but keep the music under it), set extra levels per instrument,
   and listen: original, separated, or a 30-second **remastered preview**.
4. **Export**. You get:

```
projects/<show>/output/
  Songs/01 - Song Title (Artist).flac      each song, mastered
  Stems/01 - Song Title (Artist)/          lead_vocals, backing_vocals, drums, bass, guitar,
                                           piano, woodwinds, other  (balanced exactly as in the mix)
  <show> - Full Concert.flac               the whole show, continuous, crossfaded
  <show> - Concert Vibes.flac              songs + short applause, no talking
  <show> - Full Concert.cue / Tracklist.txt
  <show> - Full Concert.srt                what the artist said, as subtitles
  Artist speech.txt, report.json
```

**Every parameter** is adjustable in **Settings** (per project, or as defaults for new projects):
hardware, every AI model, restoration, song/speech detection, speech handling, audience, song
identification, studio-reference matching, mix, mastering, export, and the full per-instrument chain
(EQ bands, expander, compression, de-esser, stereo width, reverb, target level).

## What it does, step by step

| Step | How |
| --- | --- |
| Read | ffmpeg decodes any audio/video in blocks (hours never sit in memory), SoX-quality resampling to 44.1 kHz |
| Repair | Clipped peaks (overloaded phone mic) found by their flat tops across the whole file and redrawn with splines; rumble removed |
| Crowd | **Mel-Band Roformer Crowd** separates the audience from the music |
| Vocals | Ensemble of **BS-Roformer Resurrection + Mel-Roformer Beta 6X** (averaged in the frequency domain) |
| Instruments | **BS-Roformer SW**: drums, bass, guitar, piano, other; then **MDX23C DrumSep** (kick, snare, toms, hi-hat, ride, crash) and a woodwind model (flute etc.) |
| Vocal clean-up | **Mel-Roformer de-reverb** removes the venue echo, **Mel-Roformer denoise**, then a **karaoke model** splits lead and backing vocals |
| Songs & speech | From the stems: music on/off, the crowd stem for applause, and pitch behaviour for talk vs singing (singers hold notes; speech glides). DJ/EDM sets are split where harmony and timbre change |
| Identify | Offline: your music folder and every original fetched before. Online (optional): Shazam, Whisper-transcribed lyrics, iTunes/Deezer/YouTube search. Every candidate must pass a key- and tempo-independent melody match before it's used |
| Studio tone | The original is separated the same way; each live instrument is EQ-matched to its studio counterpart and the mix balance copies the record. Nothing from the studio audio is mixed in: the performance stays 100% live |
| Per instrument | Expander (bleed and room wash down), corrective + tone EQ, level-relative compression, de-esser, mono low end, width, plate reverb on the now-dry vocal; recreated top octave for band-limited recordings (e.g. a watch) |
| Master | Glue compression, tonal balance (or the studio reference's), true-peak look-ahead limiter; −14 LUFS / −1 dBTP by default, and it stops short rather than crushing a song |

Long jobs run in chunks with crossfaded overlaps and are saved piece by piece, so a crash, a closed
lid or Stop never loses finished work.

## Quality presets and time

| Preset | Models | Use it for |
| --- | --- | --- |
| **ultra** (default) | everything above, 4 overlap passes | best result; leave it running overnight on a GPU |
| **high** | single Mel-Roformer vocal model, no denoise/kit/lead-backing, 2 passes | about 3× faster, close in quality |
| **fast** | MDX crowd model + one 6-stem pass | quick previews, CPU-only machines |

Measured on a 4-core cloud CPU (no GPU), each second of audio took about 10 s for the crowd model,
30 s for the ultra vocal ensemble, 5 s for the 6-instrument split and 5–9 s each for de-reverb,
denoise and the lead/backing split. A GPU is many times faster, but a 3-hour show on ultra is still a
job of several hours on an RTX 3060, so start it in the evening. The app shows progress and an
estimate while it runs.

## Recording tips

- **Galaxy S23 Ultra / Oppo Find X8 Ultra**: record in the highest-quality stereo mode (not "speech"
  or "interview" modes, which filter music). Video recordings (MP4) work directly.
- **Galaxy Watch 4**: watch recordings are mono and band-limited. The app widens them to stereo
  (mono-compatible) and recreates the missing top end, but the phone will always capture more detail.
- Point the mic at the stage, away from people shouting next to you, and avoid covering it.

## Honest limitations

- Separation is excellent but not perfect; very loud, distorted recordings leave artifacts,
  mostly in the guitar/piano/other stems.
- Nothing can restore what the mic never captured: deep bass a phone didn't record is rebalanced,
  not invented, and analog overload distortion can't be undone.
- An audience singing along is similar to the lead vocal and can survive crowd removal.
- YouTube sometimes asks downloaders to "sign in to confirm you're not a bot". Pick your browser in
  Settings → Song identification → *YouTube login from browser*, or give the song a file/link by hand.
  If downloads stop working, run `setup.bat` again (it updates the downloader).
- Rap is speech-like. It only counts as "talk" when the band is quiet, so rapped verses stay in songs.
- Only share recordings you have the right to share.

## Command line

```bash
concert-remaster                              # the app (same as start.bat)
concert-remaster process show.mp4 --preset high --set speech.action=remove --set master.target_lufs=-10
concert-remaster download-models              # fetch everything for offline use
concert-remaster devices                      # which GPU will be used
concert-remaster presets                      # models per preset
```

On Linux/macOS: `pip install -e ".[gpu,app]"` (or `[cpu,app]`) inside a Python 3.10–3.12
environment with ffmpeg installed.

## Models and licences

Separation runs through [python-audio-separator](https://github.com/nomadkaraoke/python-audio-separator)
(MIT) with models from the [Ultimate Vocal Remover](https://github.com/Anjok07/ultimatevocalremovergui)
community (Roformer models by viperx, KimberleyJensen, unwa, anvuew, aufr33, jarredou and others) and
[Demucs](https://github.com/facebookresearch/demucs). Speech-to-text is
[faster-whisper](https://github.com/SYSTRAN/faster-whisper) (MIT). Check each model's licence before
commercial use.

## Development

```bash
pip install -e ".[cpu,app,dev]"
pytest              # uses a fake separation backend: runs in about two minutes, no downloads
```
