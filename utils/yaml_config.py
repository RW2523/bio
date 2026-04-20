"""YAML config loading with simple `inherits` merging."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, MutableMapping

import yaml

from utils.paths import configs_dir


def _deep_merge(base: MutableMapping[str, Any], override: Mapping[str, Any]) -> MutableMapping[str, Any]:
    for k, v in override.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            _deep_merge(base[k], v)  # type: ignore[arg-type]
        else:
            base[k] = deepcopy(v)
    return base


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise TypeError(f"Expected mapping at root of {path}, got {type(data)}")
    return data


def load_merged_config(name: str, configs_root: Path | None = None) -> dict[str, Any]:
    """
    Load `configs/{name}.yaml` with optional `inherits` key (str or list of names without .yaml).

    Later files override earlier ones; the final file overrides inherited dicts shallow-deep.
    """
    root = configs_root or configs_dir()
    path = root / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Missing config: {path}")

    def load_chain(n: str, stack: set[str]) -> dict[str, Any]:
        if n in stack:
            raise ValueError(f"Inherits cycle detected involving '{n}'")
        stack.add(n)
        p = root / f"{n}.yaml"
        if not p.exists():
            raise FileNotFoundError(f"Missing inherited config: {p}")
        doc = load_yaml(p)
        inherits = doc.pop("inherits", None)
        merged: dict[str, Any] = {}
        if inherits is None:
            chain: list[str] = []
        elif isinstance(inherits, str):
            chain = [inherits]
        elif isinstance(inherits, list):
            chain = [str(x).removesuffix(".yaml") for x in inherits]
        else:
            raise TypeError(f"`inherits` must be str or list in {p}")
        for parent in chain:
            parent_name = Path(parent).stem
            _deep_merge(merged, load_chain(parent_name, stack))
        _deep_merge(merged, doc)
        stack.remove(n)
        return merged

    return load_chain(Path(name).stem, set())
