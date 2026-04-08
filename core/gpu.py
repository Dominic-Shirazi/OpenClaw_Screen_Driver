"""Centralized GPU memory management.

Provides cleanup() and get_device() so individual modules don't need
scattered try/except torch blocks for memory management.

Device priority: CUDA > MPS (Apple Silicon) > CPU.
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
            logger.debug("CUDA memory cache cleared")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            # MPS doesn't expose empty_cache yet, but calling gc helps
            import gc

            gc.collect()
            torch.mps.empty_cache()
            logger.debug("MPS memory cache cleared")
    except (ImportError, AttributeError):
        pass


def get_device(role: str) -> str:
    """Returns the configured torch device string for a role.

    Device priority: CUDA > MPS (Apple Silicon) > CPU.

    Args:
        role: One of "vlm" (detection models) or "embeddings" (CLIP, Florence-2).

    Returns:
        Device string like "cuda:0", "cuda:1", "mps", or "cpu".
    """
    try:
        import torch
    except ImportError:
        return "cpu"

    # CUDA: honour per-role GPU assignment from config
    if torch.cuda.is_available():
        cfg = get_config()
        hw = cfg.get("hardware", {})

        if role == "vlm":
            gpu_id = hw.get("gpu_vlm", 0)
        elif role == "embeddings":
            gpu_id = hw.get("gpu_embeddings", 1)
        else:
            gpu_id = 0

        return f"cuda:{gpu_id}"

    # MPS: Apple Silicon GPU acceleration
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        logger.debug("Using MPS (Apple Silicon) device")
        return "mps"

    return "cpu"
