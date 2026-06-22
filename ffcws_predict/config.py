"""Configuration loading and validation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


PUBLIC_TRUTH_URL = (
    "https://raw.githubusercontent.com/fragilefamilieschallenge/"
    "codalab-competition-bundle/master/reference2/truth.csv"
)


class ConfigError(ValueError):
    """Raised when an experiment configuration is invalid."""


def _load_mapping(path: Path) -> dict[str, Any]:
    text = path.read_text()
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(text)
    except ModuleNotFoundError:
        data = json.loads(text)
    if not isinstance(data, dict):
        raise ConfigError(f"Config must contain a mapping/object: {path}")
    return data


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).resolve()
    cfg = _load_mapping(config_path)
    cfg["_config_path"] = str(config_path)
    cfg["_config_dir"] = str(config_path.parent)
    return cfg


def get_required(cfg: dict[str, Any], dotted_key: str) -> Any:
    current: Any = cfg
    for part in dotted_key.split("."):
        if not isinstance(current, dict) or part not in current:
            raise ConfigError(f"Missing required config key: {dotted_key}")
        current = current[part]
    if current is None or current == "":
        raise ConfigError(f"Required config key is empty: {dotted_key}")
    return current


def get_optional(cfg: dict[str, Any], dotted_key: str, default: Any = None) -> Any:
    current: Any = cfg
    for part in dotted_key.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def is_url(value: str) -> bool:
    return value.startswith("http://") or value.startswith("https://")


def is_virtual_path(value: str) -> bool:
    return is_url(value) or value.startswith("zip://") or ".zip!" in value


def resolve_path(cfg: dict[str, Any], value: str | None) -> Path | str | None:
    if value is None or value == "":
        return None
    if is_virtual_path(value):
        return value
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return Path(cfg["_config_dir"]) / path


def output_dir(cfg: dict[str, Any]) -> Path:
    raw = get_required(cfg, "paths.output_dir")
    resolved = resolve_path(cfg, raw)
    assert isinstance(resolved, Path)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def target_name(cfg: dict[str, Any]) -> str:
    return str(get_optional(cfg, "target.name", "materialHardship"))


def id_column(cfg: dict[str, Any]) -> str:
    return str(get_optional(cfg, "target.id_column", "challengeID"))


def split_column(cfg: dict[str, Any]) -> str:
    return str(get_optional(cfg, "target.split_column", "split"))
