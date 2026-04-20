"""Checkpoint save/load utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path: Path, map_location: str | torch.device | None = None) -> dict[str, Any]:
    return torch.load(path, map_location=map_location, weights_only=False)
