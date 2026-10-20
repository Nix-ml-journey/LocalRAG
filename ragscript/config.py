"""Config loading. All paths in config.yaml are relative to the project root."""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
DURATIONS = [5, 10, 15, 20]


def load() -> dict:
    return yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))


def path(cfg: dict, key: str) -> Path:
    return ROOT / cfg["paths"][key]
