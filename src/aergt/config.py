from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import yaml


def read_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a YAML mapping: {path}")
    return data


def deep_update(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def load_config(path: str | Path) -> dict[str, Any]:
    cfg_path = Path(path)
    cfg = read_yaml(cfg_path)
    base_config = cfg.pop("base_config", None)
    if base_config:
        base_path = (cfg_path.parent / str(base_config)).resolve()
        cfg = deep_update(load_config(base_path), cfg)
    cfg["_config_path"] = str(cfg_path.resolve())
    return cfg


def require(cfg: Mapping[str, Any], dotted_key: str) -> Any:
    node: Any = cfg
    for part in dotted_key.split("."):
        if not isinstance(node, Mapping) or part not in node:
            raise KeyError(f"Missing required config key: {dotted_key}")
        node = node[part]
    return node


def project_root_from_config(config_path: str | Path) -> Path:
    return Path(config_path).resolve().parents[1]
