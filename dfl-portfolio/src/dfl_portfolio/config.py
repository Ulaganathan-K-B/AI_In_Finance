"""YAML configuration helpers."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    with open(path) as fh:
        cfg = yaml.safe_load(fh)
    if "base" in cfg:
        base = load_config(path.parent / cfg.pop("base"))
        cfg = deep_merge(base, cfg)
    return cfg


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out
