"""Path helpers: resolve project root and default config locations."""

from __future__ import annotations

from pathlib import Path


def project_root() -> Path:
    """Return the `project/` directory (parent of `utils/`)."""
    return Path(__file__).resolve().parents[1]


def configs_dir() -> Path:
    return project_root() / "configs"
