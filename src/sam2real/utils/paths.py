"""Project-root path helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
LEGACY_PROJECT_PREFIXES = {"sam2real-main", PROJECT_ROOT.name}


def strip_project_prefix(path: Path) -> Path:
    parts = list(path.parts)
    if parts and parts[0] in LEGACY_PROJECT_PREFIXES:
        return Path(*parts[1:]) if len(parts) > 1 else Path(".")
    return path


def resolve_project_path(value: Any, *, base_dir: Path = PROJECT_ROOT) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    path = Path(text)
    if path.is_absolute():
        return str(path)
    normalized = strip_project_prefix(path)
    return str((base_dir / normalized).resolve())


def resolve_config_path(value: Any) -> Path:
    text = str(value or "").strip()
    if not text:
        raise ValueError("Config path is empty")
    raw_path = Path(text)
    if raw_path.is_absolute():
        return raw_path
    if raw_path.exists():
        return raw_path.resolve()
    normalized = strip_project_prefix(raw_path)
    return (PROJECT_ROOT / normalized).resolve()
