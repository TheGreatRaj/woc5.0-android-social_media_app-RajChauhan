"""Turn live concert recordings into clean, studio-style masters."""

__version__ = "0.1.0"

from .pipeline import RemasterResult, RemasterSettings, remaster  # noqa: E402

__all__ = ["RemasterResult", "RemasterSettings", "remaster", "__version__"]
