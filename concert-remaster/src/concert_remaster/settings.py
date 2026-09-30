"""Every user-adjustable parameter, grouped the way the GUI shows them.

Each field carries metadata (label, help, range, choices) so the GUI can build
its controls straight from these dataclasses, and projects save them as JSON.
"""

from __future__ import annotations

import copy
import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .stem_processing import PROFILES, EQBand, StemProfile


def param(default: Any = dataclasses.MISSING, label: str = "", help: str = "", *, factory=None, **meta) -> Any:
    metadata = {"label": label, "help": help, **meta}
    if factory is not None:
        return field(default_factory=factory, metadata=metadata)
    return field(default=default, metadata=metadata)


# --- model names -------------------------------------------------------------

ROFORMER_CROWD = "mel_band_roformer_crowd_aufr33_viperx_sdr_8.7144.ckpt"
MDX_CROWD = "UVR-MDX-NET_Crowd_HQ_1.onnx"
VOCAL_ENSEMBLE = "bs_roformer_vocals_resurrection_unwa.ckpt + melband_roformer_big_beta6x.ckpt"
MEL_ROFORMER_VOCALS = "vocals_mel_band_roformer.ckpt"
BS_ROFORMER_VOCALS = "model_bs_roformer_ep_317_sdr_12.9755.ckpt"
ROFORMER_SIX_STEM = "BS-Roformer-SW.ckpt"
DEMUCS_SIX_STEM = "htdemucs_6s.yaml"
ROFORMER_DEREVERB = "dereverb_mel_band_roformer_anvuew_sdr_19.1729.ckpt"
VR_DEREVERB = "UVR-DeEcho-DeReverb.pth"
ROFORMER_DENOISE = "denoise_mel_band_roformer_aufr33_sdr_27.9959.ckpt"
KARAOKE = "mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt"
DRUMSEP = "MDX23C-DrumSep-aufr33-jarredou.ckpt"
WOODWINDS = "17_HP-Wind_Inst-UVR.pth"


@dataclass
class HardwareSettings:
    device: str = param(
        "auto", "Processing device", choices=["auto", "cuda", "directml", "cpu"],
        help="auto uses an NVIDIA GPU (CUDA) when present. 'directml' runs on AMD cards such as the RX 580 "
        "on Windows (experimental; models that fail there fall back to the CPU).",
    )
    half_precision: bool = param(False, "Half precision on NVIDIA", help="About 1.5-2x faster on CUDA with a tiny quality cost. Off = best quality.")
    model_dir: str = param("", "Model folder", help="Where AI model weights live. Empty = the 'models' folder next to the app.", kind="path")
    chunk_seconds: float = param(300.0, "Chunk length", unit="s", min=60, max=1200, step=30,
                                 help="Long recordings are separated in chunks this long. Lower it if you run out of memory.")
    chunk_overlap_seconds: float = param(10.0, "Chunk overlap", unit="s", min=2, max=30, step=1,
                                         help="Chunks overlap and crossfade by this much so no seams are audible.")


@dataclass
class ModelSettings:
    crowd_model: str = param(ROFORMER_CROWD, "Crowd removal model", kind="model", enable="crowd_enabled")
    crowd_enabled: bool = param(True, "Remove crowd noise")
    vocal_model: str = param(VOCAL_ENSEMBLE, "Vocal isolation model(s)", kind="model",
                             help="Join several models with ' + ' to average them (an ensemble): slower, cleaner.")
    vocal_ensemble_algorithm: str = param("avg_fft", "Ensemble method", choices=["avg_fft", "avg_wave", "median_wave", "max_fft", "min_fft", "uvr_max_spec", "uvr_min_spec"],
                                          help="How the vocal models are combined. avg_fft is the most balanced.")
    instrument_model: str = param(ROFORMER_SIX_STEM, "Instrument split model", kind="model",
                                  help="BS-Roformer SW splits the band into drums, bass, guitar, piano and other.")
    dereverb_model: str = param(ROFORMER_DEREVERB, "Vocal de-reverb model", kind="model", enable="dereverb_enabled")
    dereverb_enabled: bool = param(True, "Remove venue echo from vocals")
    denoise_model: str = param(ROFORMER_DENOISE, "Vocal denoise model", kind="model", enable="denoise_enabled")
    denoise_enabled: bool = param(True, "Denoise vocals")
    lead_backing_model: str = param(KARAOKE, "Lead / backing vocal model", kind="model", enable="lead_backing_enabled")
    lead_backing_enabled: bool = param(True, "Split lead and backing vocals")
    drum_kit_model: str = param(DRUMSEP, "Drum kit model", kind="model", enable="drum_kit_enabled")
    drum_kit_enabled: bool = param(True, "Split drums into kick, snare, toms, hi-hat, ride, crash")
    woodwind_model: str = param(WOODWINDS, "Woodwind model", kind="model", enable="woodwinds_enabled")
    woodwinds_enabled: bool = param(True, "Extract flute / woodwinds from 'other'")
    roformer_overlap: int = param(4, "Roformer overlap passes", min=1, max=8, step=1,
                                  help="Each audio window is processed this many times and averaged. 4 = best quality, 2 = about twice as fast.")
    demucs_shifts: int = param(2, "Demucs shifts", min=1, max=10, step=1, help="Only used if you pick a Demucs model.")


@dataclass
class RestorationSettings:
    declip: bool = param(True, "Rebuild clipped peaks")
    declip_threshold: float = param(0.98, "Clip detection level", min=0.9, max=0.999, step=0.001,
                                    help="Samples above this fraction of the ceiling, in flat-topped runs, are treated as clipped.")
    rumble_hz: float = param(25.0, "Rumble filter", unit="Hz", min=10, max=80, step=1)
    bandwidth_restore: str = param("auto", "High-frequency restoration", choices=["auto", "on", "off"],
                                   help="Recreates the top end missing from band-limited recordings (e.g. watch or voice-memo apps). "
                                   "auto turns it on only when the recording is cut off below 15 kHz.")
    bandwidth_amount: float = param(0.5, "Restoration amount", min=0, max=1, step=0.05)
    widen_mono: bool = param(True, "Widen mono recordings", help="Gives mono recordings (e.g. a watch mic) a mono-compatible stereo image.")


@dataclass
class SegmentationSettings:
    split_songs: bool = param(True, "Split the concert into songs")
    mode: str = param("auto", "Show type", choices=["auto", "breaks", "continuous"],
                      help="breaks = bands that stop between songs; continuous = DJ/EDM sets and medleys that never stop "
                      "(songs are split where the music changes); auto = split at breaks, and split any stretch longer "
                      "than the maximum song length by musical changes.")
    max_song_minutes: float = param(9.0, "Longest single song", unit="min", min=3, max=30, step=0.5)
    novelty_sensitivity: float = param(0.5, "Track-change sensitivity", min=0, max=1, step=0.05,
                                       help="For continuous sets: higher splits at smaller musical changes.")
    min_song_seconds: float = param(60.0, "Shortest song", unit="s", min=15, max=300, step=5)
    min_break_seconds: float = param(3.0, "Shortest break between songs", unit="s", min=1, max=30, step=0.5,
                                     help="Music must stop at least this long to count as the end of a song.")
    music_threshold_db: float = param(-24.0, "Music present above", unit="dB", min=-45, max=-10, step=1,
                                      help="Band level, relative to the loud parts of the show, that counts as music playing.")
    detect_speech: bool = param(True, "Detect the artist talking")
    speech_sensitivity: float = param(0.5, "Speech detection sensitivity", min=0, max=1, step=0.05,
                                      help="Higher finds more talking, but may mistake free-time singing for speech.")
    min_speech_seconds: float = param(2.0, "Shortest speech", unit="s", min=0.5, max=10, step=0.5)


@dataclass
class SpeechSettings:
    action: str = param("enhance", "When the artist talks", choices=["enhance", "keep", "remove", "music_only"],
                        help="enhance = clean up and clarify the voice; keep = leave as separated; remove = cut the talk out "
                        "(pure concert vibes); music_only = mute the voice but keep any music playing under it.")
    transcribe: bool = param(True, "Transcribe what the artist says", help="Offline Whisper speech-to-text; saved as subtitles.")
    whisper_model: str = param("large-v3", "Whisper model", choices=["tiny", "base", "small", "medium", "large-v3", "large-v3-turbo"])
    language: str = param("auto", "Language", help="auto, or a code such as hi, en, pa, ta, te, bn, mr, gu.")
    denoise: float = param(0.6, "Voice denoise", min=0, max=1, step=0.05)
    presence_db: float = param(3.0, "Presence boost", unit="dB", min=-6, max=9, step=0.5)
    warmth_db: float = param(1.0, "Warmth", unit="dB", min=-6, max=6, step=0.5)
    compression_ratio: float = param(3.0, "Voice compression", min=1, max=8, step=0.5)
    level_lu: float = param(-3.0, "Speech level vs sung vocal", unit="LU", min=-12, max=6, step=0.5)
    music_under_speech_db: float = param(-10.0, "Music under speech", unit="dB", min=-40, max=0, step=1)


@dataclass
class EffectsSettings:
    action: str = param("remove", "Stage effects", choices=["remove", "reduce", "keep"],
                        help="CO2 / smoke jets, fireworks and confetti cannons. remove = clean them out of the music, "
                        "reduce = halfway, keep = leave them in.")
    sensitivity: float = param(0.5, "Detection sensitivity", min=0, max=1, step=0.05,
                               help="Higher catches quieter effects, but may mistake a noise sweep in the music for one.")
    co2: bool = param(True, "CO2 / smoke jets")
    fireworks: bool = param(True, "Fireworks and pyro booms")
    confetti: bool = param(True, "Confetti cannons")
    level_riding: bool = param(True, "Keep the level steady",
                               help="Evens out dips and bumps (a phone's auto-gain, removed effects) without flattening builds and drops.")
    riding_range_db: float = param(4.0, "Level riding range", unit="dB", min=0, max=10, step=0.5)


@dataclass
class CrowdSettings:
    keep_in_songs: bool = param(False, "Keep audience during songs")
    in_songs_db: float = param(-18.0, "Audience level in songs", unit="LU", min=-40, max=0, step=1,
                               help="Relative to the lead vocal.")
    between_songs: str = param("shorten", "Applause between songs", choices=["keep", "shorten", "remove"])
    shorten_to_seconds: float = param(6.0, "Shorten applause to", unit="s", min=1, max=30, step=0.5)
    between_level_db: float = param(-8.0, "Applause level", unit="LU", min=-30, max=0, step=1,
                                    help="Relative to the songs' loudness.")
    fade_seconds: float = param(1.5, "Crossfade", unit="s", min=0.1, max=5, step=0.1)


@dataclass
class IdentifySettings:
    enabled: bool = param(True, "Identify songs")
    online: bool = param(True, "Look songs up online when internet is available",
                         help="Shazam + YouTube search. Everything else stays offline; found originals are cached locally.")
    artist_hint: str = param("", "Artist name", help="Helps the YouTube search, e.g. 'Arijit Singh'.")
    use_shazam: bool = param(True, "Use Shazam")
    use_lyrics_search: bool = param(True, "Search YouTube by transcribed lyrics")
    youtube_results: int = param(5, "YouTube candidates per song", min=1, max=15, step=1)
    min_match_score: float = param(0.45, "Minimum match score", min=0, max=1, step=0.01,
                                   help="How clearly a candidate's melody must follow the live song to be accepted. "
                                   "Same songs typically score 0.55+, unrelated music under 0.35.")
    cookies_browser: str = param("", "YouTube login from browser", choices=["", "edge", "chrome", "firefox", "brave", "opera"],
                                 help="If YouTube asks to 'sign in to confirm you're not a bot', pick the browser you are "
                                 "logged in with and downloads will use that login.")
    library_dir: str = param("", "Local music library", kind="path", help="Optional folder of studio tracks to match against offline.")
    reference_dir: str = param("", "Reference cache", kind="path", help="Downloaded originals are kept here. Empty = default folder.")


@dataclass
class ReferenceSettings:
    tone_match: bool = param(True, "Match the studio version's tone")
    tone_strength: float = param(0.5, "Tone match strength", min=0, max=1, step=0.05)
    tone_max_db: float = param(6.0, "Tone match limit", unit="dB", min=1, max=12, step=0.5)
    per_stem: bool = param(True, "Match each instrument separately", help="Separates the studio track too and matches stem by stem.")
    balance_match: bool = param(True, "Match the studio mix balance")
    balance_strength: float = param(0.5, "Balance match strength", min=0, max=1, step=0.05)


@dataclass
class MixSettings:
    balance_strength: float = param(0.6, "Auto-mix strength", min=0, max=1, step=0.05,
                                    help="How far stems move toward a studio balance when no reference is found.")
    max_adjust_db: float = param(6.0, "Auto-mix limit", unit="dB", min=0, max=18, step=0.5)
    ghost_cut_db: float = param(-6.0, "Empty-stem cut", unit="dB", min=-40, max=0, step=1,
                                help="Stems where the model found nothing are turned down by this much.")
    stem_gains_db: dict = param(label="Stem gains", factory=dict, kind="gains", help="Extra gain per stem after auto-mixing.")
    muted_stems: list = param(label="Muted stems", factory=list, kind="stems")
    drum_kit: bool = param(False, "Mix drums piece by piece",
                           help="Kick, snare, toms, hi-hat, ride and crash get their own mixer tracks instead of one drum track.")


@dataclass
class MasterSettings:
    target_lufs: float = param(-14.0, "Loudness", unit="LUFS", min=-24, max=-6, step=0.5, help="-14 suits streaming; -9 is loud modern pop.")
    ceiling_dbtp: float = param(-1.0, "True-peak ceiling", unit="dBTP", min=-3, max=0, step=0.1)
    max_limiting_db: float = param(6.0, "Most limiting allowed", unit="dB", min=0, max=12, step=0.5)
    glue: bool = param(True, "Bus glue compression")
    glue_depth_db: float = param(4.0, "Glue depth", unit="dB", min=0, max=15, step=0.5)
    tonal_strength: float = param(0.4, "Tonal balance EQ", min=0, max=1, step=0.05,
                                  help="Used when no studio reference was found.")
    same_loudness_all_songs: bool = param(True, "Same loudness for every song")


@dataclass
class OutputSettings:
    format: str = param("flac", "Format", choices=["flac", "wav", "mp3"])
    bit_depth: int = param(24, "Bit depth", choices=[16, 24])
    sample_rate: int = param(48000, "Sample rate", unit="Hz", choices=[44100, 48000])
    songs: bool = param(True, "One file per song")
    full_concert: bool = param(True, "Full concert (continuous)")
    vibes_edition: bool = param(True, "Concert vibes edition", help="Songs only, short applause transitions, no talking.")
    stems: str = param("per_song", "Stems", choices=["none", "per_song", "full_concert", "both"])
    transcript: bool = param(True, "Transcript of the artist's speech (.srt)")
    tracklist: bool = param(True, "Track list and cue sheet")


@dataclass
class Settings:
    hardware: HardwareSettings = field(default_factory=HardwareSettings)
    models: ModelSettings = field(default_factory=ModelSettings)
    restoration: RestorationSettings = field(default_factory=RestorationSettings)
    segmentation: SegmentationSettings = field(default_factory=SegmentationSettings)
    speech: SpeechSettings = field(default_factory=SpeechSettings)
    effects: EffectsSettings = field(default_factory=EffectsSettings)
    crowd: CrowdSettings = field(default_factory=CrowdSettings)
    identify: IdentifySettings = field(default_factory=IdentifySettings)
    reference: ReferenceSettings = field(default_factory=ReferenceSettings)
    mix: MixSettings = field(default_factory=MixSettings)
    master: MasterSettings = field(default_factory=MasterSettings)
    output: OutputSettings = field(default_factory=OutputSettings)
    stems: dict = field(default_factory=lambda: copy.deepcopy(PROFILES))


GROUP_LABELS = {
    "hardware": "Hardware",
    "models": "AI models",
    "restoration": "Restoration",
    "segmentation": "Songs & speech detection",
    "speech": "Artist speech",
    "effects": "Stage effects & level",
    "crowd": "Audience",
    "identify": "Song identification",
    "reference": "Studio reference",
    "mix": "Mix",
    "master": "Mastering",
    "output": "Export",
    "stems": "Per-instrument processing",
}


PRESETS: dict[str, dict] = {
    "ultra": {
        "description": "Every model at maximum quality. Needs an NVIDIA GPU for long shows (hours of processing).",
        "models": {},  # the defaults above
    },
    "high": {
        "description": "Top single models, 2 overlap passes: about 3x faster than ultra, close in quality.",
        "models": {
            "vocal_model": MEL_ROFORMER_VOCALS,
            "dereverb_model": VR_DEREVERB,
            "denoise_enabled": False,
            "lead_backing_enabled": False,
            "drum_kit_enabled": False,
            "roformer_overlap": 2,
        },
    },
    "fast": {
        "description": "One 6-stem pass after crowd removal. For previews or CPU-only machines.",
        "models": {
            "crowd_model": MDX_CROWD,
            "vocal_model": "",
            "dereverb_model": VR_DEREVERB,
            "denoise_enabled": False,
            "lead_backing_enabled": False,
            "drum_kit_enabled": False,
            "woodwinds_enabled": False,
            "roformer_overlap": 2,
        },
    },
}


# Sound styles set several mix/master choices at once; every value stays editable.
STYLES: dict[str, dict] = {
    "soundboard": {
        "description": "Like the mixing desk's own recording: clean and direct, audience out, gentle mastering.",
        "crowd": {"keep_in_songs": False, "between_songs": "shorten"},
        "reference": {"tone_strength": 0.5, "balance_strength": 0.5},
        "master": {"target_lufs": -14.0, "glue_depth_db": 4.0, "tonal_strength": 0.4},
        "stems": {"vocals": {"reverb": 0.08}, "lead_vocals": {"reverb": 0.08}, "backing_vocals": {"reverb": 0.12}},
    },
    "studio": {
        "description": "Closest to the released record: strong tone matching, polished and louder.",
        "crowd": {"keep_in_songs": False, "between_songs": "remove"},
        "reference": {"tone_strength": 0.85, "balance_strength": 0.8},
        "master": {"target_lufs": -11.0, "glue_depth_db": 6.0, "tonal_strength": 0.6},
        "stems": {"vocals": {"reverb": 0.12}, "lead_vocals": {"reverb": 0.12}, "backing_vocals": {"reverb": 0.18}},
    },
    "live_album": {
        "description": "A mixed live album: clean band with the audience kept in, applause between songs.",
        "crowd": {"keep_in_songs": True, "in_songs_db": -16.0, "between_songs": "keep", "between_level_db": -6.0},
        "reference": {"tone_strength": 0.5, "balance_strength": 0.5},
        "master": {"target_lufs": -12.0, "glue_depth_db": 5.0, "tonal_strength": 0.4},
        "stems": {"vocals": {"reverb": 0.15}, "lead_vocals": {"reverb": 0.15}, "backing_vocals": {"reverb": 0.2}},
    },
}


def apply_style(settings: Settings, name: str) -> Settings:
    if name not in STYLES:
        raise ValueError(f"Unknown style {name!r}; choose from {', '.join(STYLES)}")
    for group, values in STYLES[name].items():
        if group == "description":
            continue
        if group == "stems":
            for stem, changes in values.items():
                if stem in settings.stems:
                    settings.stems[stem] = dataclasses.replace(settings.stems[stem], **changes)
            continue
        setattr(settings, group, dataclasses.replace(getattr(settings, group), **values))
    return settings


def apply_preset(settings: Settings, name: str) -> Settings:
    if name not in PRESETS:
        raise ValueError(f"Unknown preset {name!r}; choose from {', '.join(PRESETS)}")
    settings.models = ModelSettings(**PRESETS[name]["models"])
    return settings


# --- serialisation -----------------------------------------------------------


def to_dict(settings: Settings) -> dict:
    data = dataclasses.asdict(settings)
    data["stems"] = {name: profile_to_dict(p) for name, p in settings.stems.items()}
    return data


def profile_to_dict(profile: StemProfile) -> dict:
    data = dataclasses.asdict(profile)
    data["eq"] = [dataclasses.asdict(b) for b in profile.eq]
    return data


def profile_from_dict(data: dict, base: StemProfile | None = None) -> StemProfile:
    base = base or StemProfile()
    values = dataclasses.asdict(base)
    values["eq"] = base.eq
    for key, value in data.items():
        if key == "eq":
            values["eq"] = tuple(EQBand(**band) for band in value)
        elif key in values:
            values[key] = value
    return StemProfile(**values)


def from_dict(data: dict | None) -> Settings:
    """Build settings from saved JSON, ignoring unknown keys and filling gaps with defaults."""
    settings = Settings()
    for f in dataclasses.fields(Settings):
        section = (data or {}).get(f.name)
        if section is None:
            continue
        if f.name == "stems":
            stems = copy.deepcopy(PROFILES)
            for name, profile in section.items():
                stems[name] = profile_from_dict(profile, PROFILES.get(name))
            settings.stems = stems
            continue
        current = getattr(settings, f.name)
        known = {g.name for g in dataclasses.fields(current)}
        setattr(settings, f.name, dataclasses.replace(current, **{k: v for k, v in section.items() if k in known}))
    return settings


def save(settings: Settings, path: str | Path) -> None:
    Path(path).write_text(json.dumps(to_dict(settings), indent=2))


def load(path: str | Path) -> Settings:
    return from_dict(json.loads(Path(path).read_text()))


def schema() -> dict:
    """Describe every parameter for the GUI: groups, labels, types, ranges, choices."""
    groups = []
    defaults = Settings()
    for f in dataclasses.fields(Settings):
        if f.name == "stems":
            continue
        section = getattr(defaults, f.name)
        params = []
        for g in dataclasses.fields(section):
            meta = dict(g.metadata)
            value = getattr(section, g.name)
            params.append({"name": g.name, "type": _type_name(value), "default": value, **meta})
        groups.append({"name": f.name, "label": GROUP_LABELS[f.name], "params": params})
    stem_params = [
        {"name": g.name, "type": _type_name(getattr(StemProfile(), g.name)), **_STEM_META.get(g.name, {"label": g.name})}
        for g in dataclasses.fields(StemProfile)
    ]
    return {
        "groups": groups,
        "stem_params": stem_params,
        "stem_defaults": {name: profile_to_dict(p) for name, p in PROFILES.items()},
        "presets": {name: p["description"] for name, p in PRESETS.items()},
        "styles": {name: st["description"] for name, st in STYLES.items()},
    }


def _type_name(value: Any) -> str:
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "str"
    if isinstance(value, (list, tuple)):
        return "list"
    if isinstance(value, dict):
        return "dict"
    return "float"  # Optional numbers default to None


_STEM_META = {
    "highpass_hz": {"label": "High-pass filter", "unit": "Hz", "min": 0, "max": 400, "step": 5, "nullable": True},
    "denoise": {"label": "Denoise", "min": 0, "max": 1, "step": 0.05},
    "expander_range_db": {"label": "Expander range", "unit": "dB below peak", "min": 10, "max": 60, "step": 1, "nullable": True,
                          "help": "Quiet bleed and room wash this far below the stem's loud parts is pushed down."},
    "expander_ratio": {"label": "Expander ratio", "min": 1, "max": 10, "step": 0.1},
    "eq": {"label": "EQ bands", "kind": "eq"},
    "compress_depth_db": {"label": "Compression depth", "unit": "dB", "min": 0, "max": 24, "step": 0.5, "nullable": True},
    "compress_ratio": {"label": "Compression ratio", "min": 1, "max": 20, "step": 0.5},
    "attack_ms": {"label": "Attack", "unit": "ms", "min": 0.1, "max": 100, "step": 0.5},
    "release_ms": {"label": "Release", "unit": "ms", "min": 10, "max": 1000, "step": 5},
    "deess": {"label": "De-esser"},
    "mono_below_hz": {"label": "Mono below", "unit": "Hz", "min": 40, "max": 300, "step": 5, "nullable": True},
    "width": {"label": "Stereo width", "min": 0, "max": 2, "step": 0.05},
    "widen_mono": {"label": "Widen if mono"},
    "reverb": {"label": "Plate reverb", "min": 0, "max": 0.5, "step": 0.01},
    "balance_db": {"label": "Target level vs vocal", "unit": "LU", "min": -20, "max": 6, "step": 0.5},
}
