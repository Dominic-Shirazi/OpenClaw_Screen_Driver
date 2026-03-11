"""Configuration loader for OCSD.

Reads config.yaml from the project root and provides cached access
via get_config(). Missing keys fall back to sensible defaults.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

_config_cache: dict | None = None

# Resolve project root relative to this file (core/config.py → project root)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_CONFIG_PATH = _PROJECT_ROOT / "config.yaml"

_DEFAULTS: dict[str, Any] = {
    "ocsd": {"version": "0.1.0"},
    "hardware": {"gpu_vlm": 0, "gpu_embeddings": 1},
    "models": {
        "vlm": "qwen2-vl",
        "vlm_host": "http://localhost:11434",
        "clip": "openai/clip-vit-base-patch32",
        "whisper_model": "small",
    },
    "paths": {
        "skills_dir": "./skills",
        "snippets_dir": "./assets/snippets",
        "faiss_index": "./assets/faiss.index",
        "replay_logs": "./logs/replays",
    },
    "execution": {
        "mouse_duration": 0.15,
        "type_interval": 0.03,
        "pixel_diff_threshold": 0.08,
        "hover_delay_ms": 300,
        "default_confidence": 0.75,
    },
    "api": {
        "host": "0.0.0.0",
        "port": 8742,
        "mcp_compatible": True,
    },
    "hub": {
        "registry_url": "https://github.com/openclaw/ocsd-hub",
        "auto_update_manifest": True,
    },
}


def _deep_merge(base: dict, override: dict) -> dict:
    """Merge override into base, recursing into nested dicts."""
    merged = base.copy()
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: str | Path | None = None) -> dict:
    """Load config from YAML file, merged with defaults.

    Args:
        path: Path to config.yaml. Defaults to project root config.yaml.
              Can be overridden with OCSD_CONFIG_PATH env var.

    Returns:
        Configuration dict with all defaults filled in.
    """
    global _config_cache

    if path is None:
        path = os.environ.get("OCSD_CONFIG_PATH", str(_DEFAULT_CONFIG_PATH))

    config_path = Path(path)
    if config_path.exists():
        with open(config_path, "r") as f:
            user_config = yaml.safe_load(f) or {}
    else:
        user_config = {}

    merged = _deep_merge(_DEFAULTS, user_config)
    _config_cache = merged
    return merged


def get_config() -> dict:
    """Get the cached config, loading from disk on first call."""
    if _config_cache is None:
        return load_config()
    return _config_cache


def reload_config() -> dict:
    """Force reload config from disk."""
    global _config_cache
    _config_cache = None
    return load_config()
