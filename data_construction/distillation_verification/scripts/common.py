#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


PROJECT_DIR = Path(__file__).resolve().parent.parent


def resolve_path(path: str | Path, base: Path = PROJECT_DIR) -> Path:
    path = Path(path)
    return path if path.is_absolute() else base / path


def load_config(config_path: str | Path) -> dict[str, Any]:
    path = resolve_path(config_path)
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}
