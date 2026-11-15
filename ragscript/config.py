"""Config loading. All paths in config.yaml are relative to the project root."""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
DURATIONS = [5, 10, 15, 20]


def load() -> dict:
    return yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))


def path(cfg: dict, key: str) -> Path:
    return ROOT / cfg["paths"][key]


def sentence_range(cfg: dict) -> str:
    """The target for average sentence length as text for prompts and reports, e.g. "10-15"."""
    lo, hi = cfg["style"]["sentence_words"]
    return f"{lo}-{hi}"
