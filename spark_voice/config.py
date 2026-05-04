"""Configuration loader – merges config.yaml with env-var overrides."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

_DEFAULT_PATH = Path(__file__).parent.parent / "config.yaml"


def load(path: str | Path | None = None) -> dict[str, Any]:
    cfg_path = Path(path) if path else _DEFAULT_PATH
    with cfg_path.open() as f:
        cfg = yaml.safe_load(f) or {}

    # Env-var overrides for secrets so they don't live in config.yaml
    _apply_env(cfg, "SPARK_API_BASE",   ["spark", "api_base"])
    _apply_env(cfg, "SPARK_API_KEY",    ["spark", "api_key"])
    _apply_env(cfg, "SPARK_MODEL",      ["spark", "model"])
    _apply_env(cfg, "HASS_URL",         ["hass", "url"])
    _apply_env(cfg, "HASS_TOKEN",       ["hass", "token"])

    return cfg


def _apply_env(cfg: dict, var: str, keys: list[str]) -> None:
    value = os.environ.get(var)
    if value is None:
        return
    node = cfg
    for k in keys[:-1]:
        node = node.setdefault(k, {})
    node[keys[-1]] = value
