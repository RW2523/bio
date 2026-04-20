"""Filesystem discovery for the WISDM directory layout."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class DiscoverySummary:
    data_root: Path
    folders: list[str] = field(default_factory=list)
    file_counts_by_ext: dict[str, int] = field(default_factory=dict)
    raw_txt_by_group: dict[str, int] = field(default_factory=dict)
    arff_by_group: dict[str, int] = field(default_factory=dict)
    activity_key_path: str | None = None
    readme_paths: list[str] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "data_root": str(self.data_root),
            "folders": self.folders,
            "file_counts_by_ext": dict(sorted(self.file_counts_by_ext.items(), key=lambda x: x[0])),
            "raw_txt_by_group": dict(sorted(self.raw_txt_by_group.items(), key=lambda x: x[0])),
            "arff_by_group": dict(sorted(self.arff_by_group.items(), key=lambda x: x[0])),
            "activity_key_path": self.activity_key_path,
            "readme_paths": self.readme_paths,
            "issues": self.issues,
        }


def discover_wisdm(data_root: Path) -> DiscoverySummary:
    if not data_root.exists():
        raise FileNotFoundError(f"data_root does not exist: {data_root}")

    summary = DiscoverySummary(data_root=data_root.resolve())
    exts: dict[str, int] = {}

    for p in data_root.rglob("*"):
        if p.is_dir():
            continue
        suf = p.suffix.lower()
        exts[suf] = exts.get(suf, 0) + 1

        rel = p.relative_to(data_root).as_posix()
        if rel.startswith("raw/") and rel.endswith(".txt"):
            # group like raw/phone/accel
            parts = rel.split("/")
            if len(parts) >= 3 and parts[0] == "raw":
                key = "/".join(parts[:3])
                summary.raw_txt_by_group[key] = summary.raw_txt_by_group.get(key, 0) + 1
        if rel.startswith("arff_files/") and rel.endswith(".arff"):
            parts = rel.split("/")
            if len(parts) >= 3 and parts[0] == "arff_files":
                key = "/".join(parts[:3])
                summary.arff_by_group[key] = summary.arff_by_group.get(key, 0) + 1

        if p.name.lower() == "activity_key.txt":
            summary.activity_key_path = rel
        if p.name.lower() in {"readme.txt", "readme.md"}:
            summary.readme_paths.append(rel)

    summary.file_counts_by_ext = exts

    # top-level folders
    summary.folders = sorted([c.name for c in data_root.iterdir() if c.is_dir()])

    raw_root = data_root / "raw"
    if not raw_root.exists():
        summary.issues.append("Expected `raw/` directory under data_root for time-series streams; not found.")

    arff_root = data_root / "arff_files"
    if not arff_root.exists():
        summary.issues.append("Expected `arff_files/` directory under data_root; not found.")

    if summary.activity_key_path is None:
        summary.issues.append("Could not find `activity_key.txt` under data_root.")

    return summary
