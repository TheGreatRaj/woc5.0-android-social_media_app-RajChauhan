# Concert Remaster

Make your concert recordings sound like the released songs, on your own PC.

- Removes crowd noise, venue echo, **CO2 jets, fireworks and confetti cannons** with local AI models,
  and keeps the level steady where they were
- Separates vocals (lead and backing), drums (and each drum), bass, guitar, keys, winds and more
- A **studio view** for every song: each instrument on its own track, with faders, pan, mute, solo
  and meters, played by the app's own audio engine
- Finds and names every song in 2–3 hour shows, including DJ/EDM sets
- Handles the artist talking: a clearer voice, subtitles, or cut out for pure concert vibes
- Matches each song's tone to its studio version (the performance stays 100% live)
- Exports mastered songs, the full show, a "concert vibes" edition, every stem, and for video
  recordings **the video with the remastered sound**, in sync

It is a desktop program: everything runs on your PC and works offline. The internet is only used,
if you allow it, to look songs up and download their original versions for comparison.

## Install (Windows 10/11, 64-bit)

1. Download this repository as a ZIP (**Code → Download ZIP**) and extract it.
2. Run **`Concert Remaster Setup.exe`**. Choose where to install (it needs about 20 GB) and your
   graphics card (or leave it on automatic). The installer then downloads Python, the AI libraries
   for your card, ffmpeg and every AI model, showing its progress. This takes a while, once.
3. Start **Concert Remaster** from the Start menu or the desktop.

Windows may say *"Windows protected your PC"* because the installer isn't code-signed: click
**More info → Run anyway**. Uninstall from *Settings → Apps* (you can keep your projects).

**Without installing:** extract the ZIP where you want the app to live, double-click `setup.bat`,
then start **`Concert Remaster.exe`**.

Full documentation: [concert-remaster/README.md](concert-remaster/README.md)
