"""Tests for core.model_cache — HuggingFace weight download utility."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestGetCacheDir:
    """Tests for get_cache_dir()."""

    def test_returns_default_when_no_config(self):
        from core.model_cache import get_cache_dir

        with patch("core.model_cache.get_config", return_value={"paths": {}}):
            result = get_cache_dir()
            assert result == Path.home() / ".cache" / "ocsd" / "models"

    def test_returns_config_value_expanded(self):
        from core.model_cache import get_cache_dir

        with patch(
            "core.model_cache.get_config",
            return_value={"paths": {"model_cache": "~/.custom/models"}},
        ):
            result = get_cache_dir()
            assert result == Path.home() / ".custom" / "models"


class TestEnsureModel:
    """Tests for ensure_model()."""

    def test_downloads_single_file(self):
        from core.model_cache import ensure_model

        mock_download = MagicMock(return_value="/fake/path/model.pt")
        with patch("huggingface_hub.hf_hub_download", mock_download), \
             patch("core.model_cache.get_cache_dir", return_value=Path("/cache")):
            result = ensure_model("microsoft/OmniParser-v2.0", "icon_detect/model.pt")

        assert result == Path("/fake/path/model.pt")
        mock_download.assert_called_once_with(
            repo_id="microsoft/OmniParser-v2.0",
            filename="icon_detect/model.pt",
            cache_dir=Path("/cache"),
        )

    def test_downloads_full_repo_when_no_filename(self):
        from core.model_cache import ensure_model

        mock_snapshot = MagicMock(return_value="/fake/repo/snapshot")
        with patch("huggingface_hub.snapshot_download", mock_snapshot), \
             patch("core.model_cache.get_cache_dir", return_value=Path("/cache")):
            result = ensure_model("microsoft/Florence-2-large")

        assert result == Path("/fake/repo/snapshot")
        mock_snapshot.assert_called_once_with(
            repo_id="microsoft/Florence-2-large",
            cache_dir=Path("/cache"),
        )

    def test_custom_cache_dir_overrides_default(self):
        from core.model_cache import ensure_model

        custom = Path("/my/cache")
        mock_download = MagicMock(return_value="/my/cache/model.pt")
        with patch("huggingface_hub.hf_hub_download", mock_download):
            ensure_model("repo/name", "file.pt", cache_dir=custom)

        mock_download.assert_called_once_with(
            repo_id="repo/name",
            filename="file.pt",
            cache_dir=custom,
        )


class TestClearCache:
    """Tests for clear_cache()."""

    def test_clear_specific_repo(self, tmp_path):
        from core.model_cache import clear_cache

        repo_dir = tmp_path / "models--microsoft--OmniParser-v2.0"
        repo_dir.mkdir(parents=True)
        (repo_dir / "model.pt").touch()

        with patch("core.model_cache.get_cache_dir", return_value=tmp_path):
            clear_cache("microsoft/OmniParser-v2.0")

        assert not repo_dir.exists()

    def test_clear_all(self, tmp_path):
        from core.model_cache import clear_cache

        repo1 = tmp_path / "models--org--repo1"
        repo1.mkdir(parents=True)
        repo2 = tmp_path / "models--org--repo2"
        repo2.mkdir(parents=True)

        with patch("core.model_cache.get_cache_dir", return_value=tmp_path):
            clear_cache()

        assert not repo1.exists()
        assert not repo2.exists()
