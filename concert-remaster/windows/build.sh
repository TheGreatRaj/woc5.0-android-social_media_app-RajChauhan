#!/usr/bin/env bash
# Builds the Windows programs from any Linux/macOS/WSL machine:
#   Concert Remaster.exe        the app launcher (no console window)
#   Concert Remaster Setup.exe  the installer (downloads everything else on the user's PC)
#
# Needs mingw-w64 (x86_64-w64-mingw32-gcc, -windres), NSIS (makensis) and git.
# Usage: concert-remaster/windows/build.sh [output folder]   (default: the repository root)
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
app="$(dirname "$here")"
root="$(dirname "$app")"
out="$(cd "${1:-$root}" && pwd)"
version="$(sed -n 's/^version = "\(.*\)"/\1/p' "$app/pyproject.toml")"
stage="$(mktemp -d)"
trap 'rm -rf "$stage"' EXIT

echo "Building Concert Remaster $version"

# 1. The launcher, with the app icon and version info
(cd "$here/launcher" && x86_64-w64-mingw32-windres launcher.rc -O coff -o "$stage/launcher.o")
x86_64-w64-mingw32-gcc -O2 -municode -mwindows -s -o "$stage/Concert Remaster.exe" \
    "$here/launcher/launcher.c" "$stage/launcher.o" -lshlwapi
cp "$stage/Concert Remaster.exe" "$out/Concert Remaster.exe"
cp "$here/launcher/app.ico" "$stage/app.ico"

# 2. The app files the installer carries (tracked files only; no tests or build tooling)
mkdir -p "$stage/concert-remaster"
(cd "$app" && git ls-files -z -- pyproject.toml README.md src windows/setup.ps1 windows/start.bat \
    | xargs -0 -I{} cp --parents {} "$stage/concert-remaster/")
# Windows line endings for the scripts
find "$stage/concert-remaster" \( -name '*.ps1' -o -name '*.bat' \) -exec sed -i 's/\r*$/\r/' {} +

# 3. The installer
makensis -V2 -DSTAGE="$stage" -DOUTFILE="$out/Concert Remaster Setup.exe" -DVERSION="$version" \
    "$here/installer/installer.nsi"

ls -l "$out/Concert Remaster.exe" "$out/Concert Remaster Setup.exe"
