"""Where the app keeps models, tools, cached references and projects.

Running from the downloaded folder (the normal Windows setup), everything
lives next to the code so the whole app is one self-contained folder:

    concert-remaster/
        models/        AI model weights
        tools/         ffmpeg, deno
        references/    studio originals found online, cached
        projects/      one folder per concert you process

Set CONCERT_REMASTER_HOME to put these somewhere else.
"""

from __future__ import annotations

import os
from pathlib import Path


def app_root() -> Path:
    env = os.environ.get("CONCERT_REMASTER_HOME")
    if env:
        return Path(env).expanduser()
    source_root = Path(__file__).resolve().parents[2]
    if (source_root / "pyproject.toml").is_file():
        return source_root
    return Path.home() / "ConcertRemaster"


def models_dir() -> Path:
    return app_root() / "models"


def tools_dir() -> Path:
    return app_root() / "tools"


def references_dir() -> Path:
    return app_root() / "references"


def projects_dir() -> Path:
    return app_root() / "projects"
