"""Command-line entry point: ``concert-remaster recording.mp4 -o out/``."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from . import __version__
from .pipeline import DEFAULT_MODEL_DIR, RemasterSettings, remaster
from .separation import PRESETS, build_plan


def _stem_gain(text: str) -> tuple[str, float]:
    name, sep, value = text.partition("=")
    if not sep:
        raise argparse.ArgumentTypeError(f"expected STEM=DB, got {text!r}")
    try:
        return name.strip().lower(), float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"gain for {name!r} must be a number of dB, got {value!r}") from None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="concert-remaster",
        description=(
            "Turn a live concert recording into a clean, studio-style master. Crowd noise and venue "
            "reverb are removed with AI models running on this machine, every instrument is split "
            "out and processed like a studio track, then the song is remixed and mastered."
        ),
    )
    parser.add_argument("inputs", nargs="*", type=Path, help="audio or video files (wav, flac, mp3, m4a, mp4, mov, ...)")
    parser.add_argument("-o", "--output", type=Path, default=Path("remastered"), help="output folder (default: ./remastered)")
    parser.add_argument("--preset", choices=sorted(PRESETS), default="balanced", help="model quality/speed trade-off (default: balanced)")
    parser.add_argument("--stems", type=int, choices=(2, 4, 6), default=6,
                        help="6 = vocals, drums, bass, guitar, piano, other; 4 = vocals, drums, bass, other; 2 = vocals + band")
    parser.add_argument("--list-presets", action="store_true", help="show the models each preset uses and exit")

    sound = parser.add_argument_group("sound")
    sound.add_argument("--target-lufs", type=float, default=-14.0, help="master loudness; -14 suits streaming, -9 is loud modern pop")
    sound.add_argument("--ceiling", type=float, default=-1.0, help="true-peak ceiling in dBTP (default: -1.0)")
    sound.add_argument("--reference", type=Path, help="a studio track whose tonal balance the master should match")
    sound.add_argument("--keep-crowd", type=float, metavar="DB",
                       help="blend the audience back in DB LU below the vocal (e.g. -15) instead of removing it")
    sound.add_argument("--gain", type=_stem_gain, action="append", default=[], metavar="STEM=DB",
                       help="extra gain for a stem after auto-mixing, e.g. --gain vocals=2 --gain drums=-1")
    sound.add_argument("--balance", type=float, default=0.6, help="how far the auto-mixer moves stems toward a studio balance, 0-1")
    sound.add_argument("--tone", type=float, default=0.5, help="strength of the mastering tonal-balance EQ, 0-1")
    sound.add_argument("--no-declip", action="store_true", help="skip rebuilding clipped peaks")

    out = parser.add_argument_group("output")
    out.add_argument("--format", choices=("wav", "flac", "mp3"), default="wav")
    out.add_argument("--bit-depth", type=int, choices=(16, 24), default=24)
    out.add_argument("--no-stems", action="store_true", help="only write the master")
    out.add_argument("--raw-stems", action="store_true", help="also write the unprocessed separated stems, crowd and removed reverb")

    models = parser.add_argument_group("models")
    models.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR, help=f"where model weights are cached (default: {DEFAULT_MODEL_DIR})")
    for stage in ("crowd", "vocals", "instruments", "dereverb", "denoise"):
        models.add_argument(f"--{stage}-model", metavar="FILE", help=f"override the {stage} model, or 'none' to skip that stage")
    models.add_argument("--overlap", type=int, metavar="N",
                        help="Roformer overlap passes (2 = fast, 4-8 = cleaner); defaults to the preset's choice")
    models.add_argument("--chunk-seconds", type=float, help="separate long recordings in chunks of this many seconds to save memory")
    models.add_argument("--keep-work-files", action="store_true", help="keep intermediate separation files for debugging")

    parser.add_argument("-v", "--verbose", action="store_true", help="show model loading and inference logs")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def _print_presets(stems: int) -> None:
    for name, preset in PRESETS.items():
        plan = build_plan(name, stems)
        print(f"{name}: {preset.description}")
        for stage, model in vars(plan).items():
            print(f"    {stage:<12} {model or '-'}")
        print()


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list_presets:
        _print_presets(args.stems)
        return 0
    if not args.inputs:
        parser.error("give at least one recording to remaster")

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    overrides = {stage: getattr(args, f"{stage}_model") for stage in ("crowd", "vocals", "instruments", "dereverb", "denoise")}
    settings = RemasterSettings(
        preset=args.preset,
        stems=args.stems,
        model_overrides={k: v for k, v in overrides.items() if v is not None},
        model_dir=args.model_dir,
        target_lufs=args.target_lufs,
        ceiling_dbtp=args.ceiling,
        crowd_db=args.keep_crowd,
        reference=args.reference,
        balance_strength=args.balance,
        stem_gains_db=dict(args.gain),
        tonal_strength=args.tone,
        declip=not args.no_declip,
        output_format=args.format,
        bit_depth=args.bit_depth,
        export_stems=not args.no_stems,
        export_raw_stems=args.raw_stems,
        chunk_seconds=args.chunk_seconds,
        roformer_overlap=args.overlap,
        keep_work_files=args.keep_work_files,
        verbose_models=args.verbose,
    )

    failures = 0
    for path in args.inputs:
        print(f"== {path}")
        try:
            result = remaster(path, args.output, settings, progress=lambda message: print(f"   {message}", flush=True))
        except Exception as exc:  # keep going with the next file
            failures += 1
            print(f"   FAILED: {exc}", file=sys.stderr)
            if args.verbose:
                raise
            continue
        m = result.report["master"]
        print(f"   Master: {result.master_path}")
        print(f"   Loudness {m['output_lufs']} LUFS, true peak {m['true_peak_dbtp']} dBTP")
        if result.stem_paths:
            print(f"   Stems:  {result.master_path.parent / 'stems'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
