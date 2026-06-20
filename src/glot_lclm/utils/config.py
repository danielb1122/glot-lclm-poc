from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if cfg is None:
        raise ValueError(f"Empty config: {path}")
    return cfg


def save_config(config: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)


def _parse_scalar(value: str) -> Any:
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"null", "none"}:
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    if value.startswith("[") or value.startswith("{"):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def apply_overrides(config: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    """Apply CLI overrides like ``compression.ratio=4`` to a config copy."""
    out = copy.deepcopy(config)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override must be key=value, got: {item}")
        key, raw_value = item.split("=", 1)
        parts = key.split(".")
        cursor: Any = out
        for idx, part in enumerate(parts[:-1]):
            next_part = parts[idx + 1]
            if isinstance(cursor, list):
                if not part.isdigit():
                    raise ValueError(f"Expected list index in override {key!r}, got {part!r}")
                list_idx = int(part)
                cursor = cursor[list_idx]
                continue

            if not isinstance(cursor, dict):
                raise ValueError(f"Cannot descend into non-container for override {key!r}")

            if part not in cursor or cursor[part] is None:
                cursor[part] = [] if next_part.isdigit() else {}
            cursor = cursor[part]

        leaf = parts[-1]
        value = _parse_scalar(raw_value)
        if isinstance(cursor, list):
            if not leaf.isdigit():
                raise ValueError(f"Expected list index in override {key!r}, got {leaf!r}")
            cursor[int(leaf)] = value
        else:
            cursor[leaf] = value
    return out


def flatten_dict(d: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in d.items():
        new_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(flatten_dict(value, new_key))
        else:
            flat[new_key] = value
    return flat


def get_by_path(config: dict[str, Any], path: str, default: Any = None) -> Any:
    cursor: Any = config
    for part in path.split("."):
        if not isinstance(cursor, dict) or part not in cursor:
            return default
        cursor = cursor[part]
    return cursor
