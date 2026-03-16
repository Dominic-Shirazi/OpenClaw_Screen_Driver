"""Tests for the TUI launcher (recorder.tui)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_skill_files(tmp_path: Path, names: list[str]) -> Path:
    """Creates fake skill JSON files in a temp directory.

    Args:
        tmp_path: Pytest tmp_path fixture.
        names: List of filenames (without extension) to create.

    Returns:
        The directory containing the created files.
    """
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    for name in names:
        fp = skills_dir / f"{name}.json"
        fp.write_text(json.dumps({"name": name, "nodes": []}))
    return skills_dir


# ---------------------------------------------------------------------------
# Import guard
# ---------------------------------------------------------------------------


class TestRichImportGuard:
    """Verifies graceful failure when rich is not installed."""

    def test_import_error_without_rich(self) -> None:
        """launch_menu raises ImportError with a helpful message when
        rich is missing."""
        # Temporarily remove rich from sys.modules and block re-import
        saved_modules: dict[str, Any] = {}
        rich_keys = [k for k in sys.modules if k == "rich" or k.startswith("rich.")]
        for k in rich_keys:
            saved_modules[k] = sys.modules.pop(k)

        # Also remove our own module so it re-imports
        tui_key = "recorder.tui"
        saved_tui = sys.modules.pop(tui_key, None)

        import builtins

        _real_import = builtins.__import__

        def _mock_import(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "rich" or name.startswith("rich."):
                raise ImportError("No module named 'rich'")
            return _real_import(name, *args, **kwargs)

        try:
            with patch("builtins.__import__", side_effect=_mock_import):
                with pytest.raises(ImportError, match="rich"):
                    import importlib

                    importlib.import_module("recorder.tui")
        finally:
            # Restore everything
            sys.modules.update(saved_modules)
            if saved_tui is not None:
                sys.modules[tui_key] = saved_tui


# ---------------------------------------------------------------------------
# Menu flow tests
# ---------------------------------------------------------------------------


class TestLaunchMenu:
    """Tests for launch_menu() with mocked rich prompts."""

    def test_record_returns_command_and_skill_name(self) -> None:
        """Selecting 'Record Workflow' prompts for a name and returns
        ('record', {'skill_name': ...})."""
        with (
            patch("recorder.tui.Prompt.ask", side_effect=["1", "login_flow"]),
            patch("recorder.tui.Console"),
        ):
            from recorder.tui import launch_menu

            cmd, kwargs = launch_menu()

        assert cmd == "record"
        assert kwargs == {"skill_name": "login_flow"}

    def test_diagram_returns_command_and_skill_name(self) -> None:
        """Selecting 'Annotate Diagram' prompts for a name and returns
        ('diagram', {'skill_name': ...})."""
        with (
            patch("recorder.tui.Prompt.ask", side_effect=["2", "chrome_home"]),
            patch("recorder.tui.Console"),
        ):
            from recorder.tui import launch_menu

            cmd, kwargs = launch_menu()

        assert cmd == "diagram"
        assert kwargs == {"skill_name": "chrome_home"}

    def test_execute_with_file_picker(self, tmp_path: Path) -> None:
        """Selecting 'Execute Skill' shows a file picker when skills
        exist and returns ('execute', {'skill_file': ...})."""
        skills_dir = _make_skill_files(tmp_path, ["login", "checkout"])

        with (
            patch("recorder.tui.Prompt.ask", side_effect=["3"]),
            patch("recorder.tui.IntPrompt.ask", return_value=1),
            patch("recorder.tui.Console"),
        ):
            from recorder.tui import launch_menu

            cmd, kwargs = launch_menu(skills_dir=skills_dir)

        assert cmd == "execute"
        assert "skill_file" in kwargs
        assert "checkout.json" in kwargs["skill_file"]

    def test_execute_manual_path_when_no_skills(self, tmp_path: Path) -> None:
        """When no skill files exist, prompts for a manual path."""
        empty_dir = tmp_path / "empty_skills"
        empty_dir.mkdir()

        with (
            patch(
                "recorder.tui.Prompt.ask",
                side_effect=["3", "path/to/my_skill.json"],
            ),
            patch("recorder.tui.Console"),
        ):
            from recorder.tui import launch_menu

            cmd, kwargs = launch_menu(skills_dir=empty_dir)

        assert cmd == "execute"
        assert kwargs["skill_file"] == "path/to/my_skill.json"

    def test_compose_returns_skill_file(self, tmp_path: Path) -> None:
        """Selecting 'Connect Steps' picks a skill file and returns
        ('compose', {'skill_file': ...})."""
        skills_dir = _make_skill_files(tmp_path, ["diagram_v1"])

        with (
            patch("recorder.tui.Prompt.ask", side_effect=["4"]),
            patch("recorder.tui.IntPrompt.ask", return_value=1),
            patch("recorder.tui.Console"),
        ):
            from recorder.tui import launch_menu

            cmd, kwargs = launch_menu(skills_dir=skills_dir)

        assert cmd == "compose"
        assert "diagram_v1.json" in kwargs["skill_file"]

    def test_help_returns_empty_kwargs(self) -> None:
        """Selecting 'Help' returns ('help', {})."""
        with (
            patch("recorder.tui.Prompt.ask", side_effect=["5"]),
            patch("recorder.tui.Console"),
        ):
            from recorder.tui import launch_menu

            cmd, kwargs = launch_menu()

        assert cmd == "help"
        assert kwargs == {}


# ---------------------------------------------------------------------------
# Skill scanner tests
# ---------------------------------------------------------------------------


class TestScanSkills:
    """Tests for _scan_skills helper."""

    def test_returns_sorted_json_files(self, tmp_path: Path) -> None:
        """_scan_skills returns sorted list of .json paths."""
        skills_dir = _make_skill_files(tmp_path, ["zebra", "alpha", "middle"])

        from recorder.tui import _scan_skills

        result = _scan_skills(skills_dir)

        names = [p.stem for p in result]
        assert names == ["alpha", "middle", "zebra"]

    def test_returns_empty_for_missing_dir(self, tmp_path: Path) -> None:
        """_scan_skills returns [] when directory does not exist."""
        from recorder.tui import _scan_skills

        result = _scan_skills(tmp_path / "nonexistent")
        assert result == []

    def test_ignores_non_json(self, tmp_path: Path) -> None:
        """_scan_skills only returns .json files."""
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        (skills_dir / "readme.txt").write_text("not a skill")
        (skills_dir / "real.json").write_text("{}")

        from recorder.tui import _scan_skills

        result = _scan_skills(skills_dir)
        assert len(result) == 1
        assert result[0].name == "real.json"
