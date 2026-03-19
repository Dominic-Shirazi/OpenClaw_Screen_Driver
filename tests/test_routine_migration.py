"""Tests for routine migration (v0 -> v1) and discovery modules."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest


def _make_v0_routine(
    name: str = "test-routine",
    num_steps: int = 2,
) -> dict[str, Any]:
    """Build a minimal v0 routine dict for testing."""
    steps = []
    for i in range(num_steps):
        node_id = f"node-{i:04d}"
        steps.append(
            {
                "step_index": i,
                "node_id": node_id,
                "element_type": "button",
                "label": f"Step {i}",
                "caption": "",
                "action": "click",
                "bbox": {"x": 100, "y": 200, "w": 50, "h": 30},
                "bbox_pct": {
                    "x_pct": 100 / 1920,
                    "y_pct": 200 / 1080,
                    "w_pct": 50 / 1920,
                    "h_pct": 30 / 1080,
                },
                "region_hint": "top_left",
                "snippet_path": f"snippets/step_{i:02d}.png",
                "embedding_path": f"embeddings/step_{i:02d}.npy",
                "confidence": 0.95,
                "dry_run_passed": True,
            }
        )

    return {
        "$schema": "ocsd-routine-v0",
        "name": name,
        "description": "A test routine",
        "start_from": "desktop",
        "created_at": "2026-03-18T12:00:00+00:00",
        "resolution": [1920, 1080],
        "steps": steps,
        "graph": {"nodes": {}, "edges": []},
    }


# ── detect_schema_version ──────────────────────────────────────────


class TestDetectSchemaVersion:
    """Tests for detect_schema_version."""

    def test_detects_v0(self) -> None:
        from routine.migration import detect_schema_version

        data = {"$schema": "ocsd-routine-v0"}
        assert detect_schema_version(data) == "v0"

    def test_detects_v1(self) -> None:
        from routine.migration import detect_schema_version

        data = {"$schema": "ocsd-routine-v1"}
        assert detect_schema_version(data) == "v1"

    def test_returns_unknown_for_missing(self) -> None:
        from routine.migration import detect_schema_version

        assert detect_schema_version({}) == "unknown"

    def test_returns_unknown_for_unrecognized(self) -> None:
        from routine.migration import detect_schema_version

        assert detect_schema_version({"$schema": "other-thing"}) == "unknown"


# ── upgrade_v0_to_v1 ───────────────────────────────────────────────


class TestUpgradeV0ToV1:
    """Tests for upgrade_v0_to_v1."""

    def test_sets_schema_to_v1(self) -> None:
        from routine.migration import upgrade_v0_to_v1

        v0 = _make_v0_routine()
        v1 = upgrade_v0_to_v1(v0)
        assert v1["$schema"] == "ocsd-routine-v1"

    def test_adds_missing_v1_fields(self) -> None:
        from routine.migration import upgrade_v0_to_v1

        v0 = _make_v0_routine()
        v1 = upgrade_v0_to_v1(v0)

        assert "category" in v1
        assert "tags" in v1
        assert "programs" in v1
        assert "platform" in v1
        assert "theme" in v1
        assert "author" in v1
        assert "author_display" in v1
        assert "version" in v1
        assert "signature" in v1
        assert "checksum" in v1

    def test_converts_snippet_path_to_node_id(self) -> None:
        from routine.migration import upgrade_v0_to_v1

        v0 = _make_v0_routine()
        v1 = upgrade_v0_to_v1(v0)

        for step in v1["steps"]:
            node_id = step["node_id"]
            assert step["snippet_path"] == f"snippets/{node_id}.png"

    def test_converts_embedding_path_to_node_id(self) -> None:
        from routine.migration import upgrade_v0_to_v1

        v0 = _make_v0_routine()
        v1 = upgrade_v0_to_v1(v0)

        for step in v1["steps"]:
            node_id = step["node_id"]
            assert step["embedding_path"] == f"embeddings/{node_id}.npy"

    def test_adds_anchors_to_each_step(self) -> None:
        from routine.migration import upgrade_v0_to_v1

        v0 = _make_v0_routine()
        v1 = upgrade_v0_to_v1(v0)

        for step in v1["steps"]:
            assert "anchors" in step
            anchors = step["anchors"]
            assert "visual_match" in anchors
            assert "ocr_text" in anchors
            assert "position_pct" in anchors
            assert "region_hint" in anchors

    def test_preserves_existing_fields(self) -> None:
        from routine.migration import upgrade_v0_to_v1

        v0 = _make_v0_routine(name="my-routine")
        v1 = upgrade_v0_to_v1(v0)

        assert v1["name"] == "my-routine"
        assert v1["description"] == "A test routine"
        assert v1["start_from"] == "desktop"
        assert v1["created_at"] == "2026-03-18T12:00:00+00:00"
        assert v1["resolution"] == [1920, 1080]
        assert len(v1["steps"]) == 2
        assert "graph" in v1

    def test_does_not_mutate_input(self) -> None:
        from routine.migration import upgrade_v0_to_v1

        v0 = _make_v0_routine()
        original_schema = v0["$schema"]
        upgrade_v0_to_v1(v0)
        assert v0["$schema"] == original_schema

    def test_sets_updated_at_from_created_at(self) -> None:
        from routine.migration import upgrade_v0_to_v1

        v0 = _make_v0_routine()
        v1 = upgrade_v0_to_v1(v0)
        assert v1["updated_at"] == v0["created_at"]


# ── discovery ──────────────────────────────────────────────────────


class TestGetRoutineDir:
    """Tests for get_routine_dir."""

    def test_returns_expanded_path(self) -> None:
        from routine.discovery import get_routine_dir

        result = get_routine_dir()
        assert result == Path.home() / ".ocsd" / "routines"
        assert result.is_absolute()


class TestListRoutines:
    """Tests for list_routines."""

    def test_returns_empty_for_nonexistent_dir(self, tmp_path: Path) -> None:
        from routine.discovery import list_routines

        result = list_routines(tmp_path / "nonexistent")
        assert result == []

    def test_lists_routines_with_routine_json(self, tmp_path: Path) -> None:
        from routine.discovery import list_routines

        # Create two routine directories
        for name in ["alpha", "beta"]:
            d = tmp_path / name
            d.mkdir()
            (d / "routine.json").write_text(
                json.dumps({"$schema": "ocsd-routine-v1", "name": name}),
                encoding="utf-8",
            )

        result = list_routines(tmp_path)
        assert len(result) == 2
        names = [r.name for r in result]
        assert "alpha" in names
        assert "beta" in names

    def test_ignores_dirs_without_routine_json(self, tmp_path: Path) -> None:
        from routine.discovery import list_routines

        # Directory with routine.json
        good = tmp_path / "good"
        good.mkdir()
        (good / "routine.json").write_text(
            json.dumps({"$schema": "ocsd-routine-v1", "name": "good"}),
            encoding="utf-8",
        )

        # Directory without routine.json
        bad = tmp_path / "bad"
        bad.mkdir()
        (bad / "something.txt").write_text("nope", encoding="utf-8")

        result = list_routines(tmp_path)
        assert len(result) == 1
        assert result[0].name == "good"

    def test_routine_info_has_schema_version(self, tmp_path: Path) -> None:
        from routine.discovery import list_routines

        d = tmp_path / "routine_a"
        d.mkdir()
        (d / "routine.json").write_text(
            json.dumps({"$schema": "ocsd-routine-v0", "name": "routine_a"}),
            encoding="utf-8",
        )

        result = list_routines(tmp_path)
        assert result[0].schema_version == "v0"
        assert result[0].path == d
