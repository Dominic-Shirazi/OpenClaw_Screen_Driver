"""HuggingFace model weight download and cache management.

Provides a single utility for downloading model weights from HuggingFace Hub
and caching them locally. Used by OmniParser and Florence-2 modules.

No GPU, no torch — pure download utility.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from core.config import get_config

logger = logging.getLogger(__name__)


def get_cache_dir() -> Path:
    """Return model cache directory from config or default ~/.cache/ocsd/models/."""
    cfg = get_config()
    custom = cfg.get("paths", {}).get("model_cache")
    if custom:
        return Path(custom).expanduser()
    return Path.home() / ".cache" / "ocsd" / "models"


def ensure_model(
    repo_id: str,
    filename: str | None = None,
    cache_dir: Path | None = None,
) -> Path:
    """Download model file from HuggingFace if not cached. Return local path.

    Args:
        repo_id: HuggingFace repo (e.g., "microsoft/OmniParser-v2.0").
        filename: Specific file within repo (e.g., "icon_detect/model.pt").
                  If None, downloads entire repo snapshot.
        cache_dir: Override default cache directory.

    Returns:
        Path to local cached file or directory.
    """
    from huggingface_hub import hf_hub_download, snapshot_download

    if cache_dir is None:
        cache_dir = get_cache_dir()

    if filename is not None:
        logger.info("Ensuring model file: %s/%s", repo_id, filename)
        local_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            cache_dir=cache_dir,
        )
        return Path(local_path)
    else:
        logger.info("Ensuring model repo: %s", repo_id)
        local_dir = snapshot_download(
            repo_id=repo_id,
            cache_dir=cache_dir,
        )
        return Path(local_dir)


def clear_cache(repo_id: str | None = None) -> None:
    """Remove cached weights. If repo_id given, only that repo.

    Args:
        repo_id: HuggingFace repo to clear (e.g., "microsoft/OmniParser-v2.0").
                 If None, removes all cached model directories.
    """
    cache_dir = get_cache_dir()
    if not cache_dir.exists():
        return

    if repo_id is not None:
        # HuggingFace cache uses models--org--repo format
        dir_name = f"models--{repo_id.replace('/', '--')}"
        target = cache_dir / dir_name
        if target.exists():
            shutil.rmtree(target)
            logger.info("Cleared cache for %s", repo_id)
    else:
        for child in cache_dir.iterdir():
            if child.is_dir() and child.name.startswith("models--"):
                shutil.rmtree(child)
        logger.info("Cleared all model caches")
