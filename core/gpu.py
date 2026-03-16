"""Centralized GPU memory management.

Provides cleanup() and get_device() so individual modules don't need
scattered try/except torch blocks for memory management.
"""

from __future__ import annotations

import logging

from core.config import get_config

logger = logging.getLogger(__name__)


def cleanup() -> None:
    """Free GPU memory caches. Safe to call even without torch."""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.debug("GPU memory cache cleared")
    except ImportError:
        pass


def get_device(role: str) -> str:
    """Returns the configured torch device string for a role.

    Args:
        role: One of "vlm" (detection models) or "embeddings" (CLIP, Florence-2).

    Returns:
        Device string like "cuda:0", "cuda:1", or "cpu".
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return "cpu"
    except ImportError:
        return "cpu"

    cfg = get_config()
    hw = cfg.get("hardware", {})

    if role == "vlm":
        gpu_id = hw.get("gpu_vlm", 0)
    elif role == "embeddings":
        gpu_id = hw.get("gpu_embeddings", 1)
    else:
        gpu_id = 0

    return f"cuda:{gpu_id}"
