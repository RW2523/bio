"""Small helpers for atomic-ish cache writes."""

from __future__ import annotations

import os
from pathlib import Path


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def replace_atomic(tmp_path: Path, final_path: Path) -> None:
    """Best-effort atomic replace (same filesystem)."""
    os.replace(tmp_path, final_path)
