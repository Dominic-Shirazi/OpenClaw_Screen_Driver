"""Tests for routine management operations: fork, delete, inspect, and version helpers.

Covers: MGMT-02 (fork), MGMT-03 (delete), MGMT-04 (inspect),
version bumping helpers (bump_minor, bump_patch).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from mapper.graph import OCSDGraph
from routine.format import Routine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_test_routine(
    base_dir: Path,
    name: str = "test-routine",
    num_steps: int = 3,
) -> Path:
    """Create a minimal valid routine directory for testing.

    Args:
        base_dir: Parent directory to create routine folder in.
        name: Routine directory name.
        num_steps: Number of dummy steps to create.

    Returns:
        Path to the routine directory.
    """
    routine_dir = base_dir / name
    routine_dir.mkdir(parents=True, exist_ok=True)

    # Build steps with node_ids and snippet/embedding paths
    steps: list[dict[str, Any]] = []
    graph = OCSDGraph()
    for i in range(num_steps):
        node_id = f"node-{i:03d}"
        graph._graph.add_node(node_id, node_id=node_id)
        steps.append({
            "step_index": i,
            "node_id": node_id,
            "element_type": "button",
            "label": f"Step {i}",
            "caption": f"Caption {i}",
            "confidence": 0.9,
            "action": "click",
            "bbox": {"x": 100 + i * 10, "y": 200, "w": 50, "h": 30},
            "bbox_pct": {"x_pct": 0.05, "y_pct": 0.18, "w_pct": 0.03, "h_pct": 0.03},
            "region_hint": "top_left",
            "anchors": {
                "visual_match": f"snippets/{node_id}.png",
                "ocr_text": f"Step {i}",
                "position_pct": {"x_pct": 0.07, "y_pct": 0.2},
                "region_hint": "top_left",
            },
            "snippet_path": f"snippets/{node_id}.png",
            "embedding_path": f"embeddings/{node_id}.npy",
            "dry_run_passed": True,
        })

        # Create dummy snippet and embedding files
        snippets_dir = routine_dir / "snippets"
        snippets_dir.mkdir(exist_ok=True)
        (snippets_dir / f"{node_id}.png").write_bytes(b"fake-png")

        embeddings_dir = routine_dir / "embeddings"
        embeddings_dir.mkdir(exist_ok=True)
        (embeddings_dir / f"{node_id}.npy").write_bytes(b"fake-npy")

    routine = Routine(
        name=name,
        version="1.2.3",
        steps=steps,
        graph=graph,
    )
    routine.save(routine_dir)
    return routine_dir


# ---------------------------------------------------------------------------
# Version helpers
# ---------------------------------------------------------------------------


class TestBumpMinor:
    """Tests for bump_minor version helper."""

    def test_bump_minor_increments_minor(self) -> None:
        """bump_minor('1.2.3') returns '1.3.0'."""
        from routine.version import bump_minor

        assert bump_minor("1.2.3") == "1.3.0"

    def test_bump_minor_from_zero(self) -> None:
        """bump_minor('1.0.0') returns '1.1.0'."""
        from routine.version import bump_minor

        assert bump_minor("1.0.0") == "1.1.0"


class TestBumpPatch:
    """Tests for bump_patch version helper."""

    def test_bump_patch_increments_patch(self) -> None:
        """bump_patch('1.2.3') returns '1.2.4'."""
        from routine.version import bump_patch

        assert bump_patch("1.2.3") == "1.2.4"

    def test_bump_patch_from_zero(self) -> None:
        """bump_patch('1.0.0') returns '1.0.1'."""
        from routine.version import bump_patch

        assert bump_patch("1.0.0") == "1.0.1"


# ---------------------------------------------------------------------------
# Fork routine
# ---------------------------------------------------------------------------


class TestForkRoutine:
    """Tests for fork_routine management function."""

    def test_fork_creates_copy(self, tmp_path: Path) -> None:
        """fork_routine creates target directory with all files copied."""
        from routine.management import fork_routine

        source = _make_test_routine(tmp_path, "original", num_steps=2)
        target = fork_routine(source, "forked-copy", base_dir=tmp_path)

        assert target.exists()
        assert (target / "routine.json").exists()
        assert (target / "snippets" / "node-000.png").exists()
        assert (target / "snippets" / "node-001.png").exists()

    def test_fork_resets_name_and_version(self, tmp_path: Path) -> None:
        """fork_routine resets version to 1.0.0 and sets new name."""
        from routine.management import fork_routine

        source = _make_test_routine(tmp_path, "src", num_steps=1)
        target = fork_routine(source, "new-name", base_dir=tmp_path)

        loaded = Routine.load(target)
        assert loaded.name == "new-name"
        assert loaded.version == "1.0.0"

    def test_fork_with_truncation(self, tmp_path: Path) -> None:
        """fork_routine with truncate_at=2 keeps only steps 0,1."""
        from routine.management import fork_routine

        source = _make_test_routine(tmp_path, "source-trunc", num_steps=4)
        target = fork_routine(source, "truncated", truncate_at=2, base_dir=tmp_path)

        loaded = Routine.load(target)
        assert len(loaded.steps) == 2
        assert loaded.steps[0]["node_id"] == "node-000"
        assert loaded.steps[1]["node_id"] == "node-001"

        # Orphaned snippet/embedding files should be removed
        assert (target / "snippets" / "node-000.png").exists()
        assert (target / "snippets" / "node-001.png").exists()
        assert not (target / "snippets" / "node-002.png").exists()
        assert not (target / "snippets" / "node-003.png").exists()
        assert not (target / "embeddings" / "node-002.npy").exists()
        assert not (target / "embeddings" / "node-003.npy").exists()

    def test_fork_raises_if_target_exists(self, tmp_path: Path) -> None:
        """fork_routine raises ValueError if target name already exists."""
        from routine.management import fork_routine

        source = _make_test_routine(tmp_path, "src2", num_steps=1)
        (tmp_path / "already-exists").mkdir()

        with pytest.raises(ValueError, match="already exists"):
            fork_routine(source, "already-exists", base_dir=tmp_path)


# ---------------------------------------------------------------------------
# Delete routine
# ---------------------------------------------------------------------------


class TestDeleteRoutine:
    """Tests for delete_routine management function."""

    def test_delete_with_confirm_yes(self, tmp_path: Path) -> None:
        """delete_routine with confirm=yes removes directory, returns True."""
        from routine.management import delete_routine

        routine_dir = _make_test_routine(tmp_path, "to-delete", num_steps=1)
        assert routine_dir.exists()

        result = delete_routine(routine_dir, skip_confirm=True)

        assert result is True
        assert not routine_dir.exists()

    def test_delete_with_confirm_no(self, tmp_path: Path) -> None:
        """delete_routine with user declining keeps directory, returns False."""
        from routine.management import delete_routine

        routine_dir = _make_test_routine(tmp_path, "to-keep", num_steps=1)

        with patch("routine.management.Confirm.ask", return_value=False):
            result = delete_routine(routine_dir, skip_confirm=False)

        assert result is False
        assert routine_dir.exists()


# ---------------------------------------------------------------------------
# Inspect routine
# ---------------------------------------------------------------------------


class TestInspectRoutine:
    """Tests for inspect_routine management function."""

    def test_inspect_as_json(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        """inspect_routine with as_json=True outputs valid JSON."""
        from routine.management import inspect_routine

        routine_dir = _make_test_routine(tmp_path, "json-inspect", num_steps=2)
        inspect_routine(routine_dir, as_json=True)

        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["name"] == "json-inspect"
        assert len(data["steps"]) == 2

    def test_inspect_rich_table(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        """inspect_routine with as_json=False outputs Rich table with step data."""
        from routine.management import inspect_routine

        routine_dir = _make_test_routine(tmp_path, "table-inspect", num_steps=2)
        inspect_routine(routine_dir, as_json=False)

        captured = capsys.readouterr()
        # Rich table output should contain step labels
        assert "Step 0" in captured.out
        assert "Step 1" in captured.out
