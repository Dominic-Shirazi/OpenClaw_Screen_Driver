"""Tests for routine file format — ocsd-routine-v1.

Covers: FMT-01 (schema structure), FMT-02 (VLM metadata in steps),
FMT-05 (graph round-trip), FMT-07 (metadata fields).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from mapper.graph import OCSDGraph
from routine.checksum import calculate_routine_checksum
from routine.format import VALID_CATEGORIES, Routine, build_v1_step


# ---------------------------------------------------------------------------
# FMT-01: Schema structure
# ---------------------------------------------------------------------------


def test_to_dict_has_schema_v1() -> None:
    """Routine.to_dict() returns dict with $schema == ocsd-routine-v1."""
    r = Routine(name="test")
    d = r.to_dict()
    assert d["$schema"] == "ocsd-routine-v1"


def test_to_dict_key_order() -> None:
    """to_dict() key order matches the spec exactly."""
    r = Routine(name="test")
    d = r.to_dict()
    expected_keys = [
        "$schema", "name", "version", "description", "author", "author_display",
        "created_at", "updated_at", "category", "tags", "programs", "platform",
        "theme", "start_from", "resolution", "steps", "graph", "checksum",
        "signature",
    ]
    assert list(d.keys()) == expected_keys


def test_json_output_formatting() -> None:
    """JSON output uses 2-space indent and ensure_ascii=False."""
    r = Routine(name="test")
    d = r.to_dict()
    output = json.dumps(d, indent=2, ensure_ascii=False)
    # Check 2-space indent is used (first nested key)
    lines = output.split("\n")
    # Second line should start with exactly 2 spaces
    assert lines[1].startswith("  ")
    assert not lines[1].startswith("    ")


# ---------------------------------------------------------------------------
# FMT-02: VLM metadata in steps
# ---------------------------------------------------------------------------


def test_step_dict_keys() -> None:
    """Step dict contains required keys including VLM metadata."""
    step = {
        "tag_data": {
            "element_type": "button",
            "label": "Submit",
            "caption": "Submit form button",
            "action": "click",
            "confidence": 0.95,
            "ocr_text": "Submit",
        },
        "bbox": (100, 200, 50, 30),
    }
    result = build_v1_step(step, index=0, node_id="node-001", screen_w=1920, screen_h=1080)

    expected_keys = {
        "step_index", "node_id", "element_type", "label", "caption",
        "confidence", "action", "bbox", "bbox_pct", "region_hint",
        "anchors", "snippet_path", "embedding_path", "dry_run_passed",
    }
    assert set(result.keys()) == expected_keys


def test_step_anchors_keys() -> None:
    """Step anchors dict contains required anchor keys."""
    step = {
        "tag_data": {
            "element_type": "button",
            "label": "OK",
            "caption": "Confirm",
            "action": "click",
            "ocr_text": "OK",
        },
        "bbox": (960, 540, 100, 50),
    }
    result = build_v1_step(step, index=0, node_id="node-002", screen_w=1920, screen_h=1080)
    anchors = result["anchors"]

    expected_anchor_keys = {"visual_match", "ocr_text", "position_pct", "region_hint"}
    assert set(anchors.keys()) == expected_anchor_keys


# ---------------------------------------------------------------------------
# FMT-05: Checksum
# ---------------------------------------------------------------------------


def test_checksum_is_sha256_hex_64() -> None:
    """Checksum is SHA256 hex string of length 64."""
    steps = [{"step_index": 0, "node_id": "a"}]
    graph_dict = {"skill_id": "", "nodes": [], "edges": []}
    result = calculate_routine_checksum(steps, graph_dict)
    assert len(result) == 64
    assert all(c in "0123456789abcdef" for c in result)


def test_checksum_stable_across_metadata_changes() -> None:
    """Checksum does NOT change when name, description, tags, or updated_at change."""
    r1 = Routine(name="alpha", description="first", tags=["a"])
    r2 = Routine(name="beta", description="second", tags=["b", "c"])
    r2.updated_at = "2099-01-01T00:00:00+00:00"

    # Same steps and graph -> same checksum
    d1 = r1.to_dict()
    d2 = r2.to_dict()
    assert d1["checksum"] == d2["checksum"]


def test_checksum_changes_with_steps() -> None:
    """Checksum DOES change when steps change."""
    r1 = Routine(name="test")
    r2 = Routine(name="test")
    r2.steps = [{"step_index": 0, "node_id": "xyz", "element_type": "button"}]

    d1 = r1.to_dict()
    d2 = r2.to_dict()
    assert d1["checksum"] != d2["checksum"]


def test_checksum_changes_with_graph() -> None:
    """Checksum DOES change when graph data changes."""
    r1 = Routine(name="test")

    r2 = Routine(name="test")
    g = OCSDGraph()
    g.add_node("button", "Click Me")
    r2.graph = g

    d1 = r1.to_dict()
    d2 = r2.to_dict()
    assert d1["checksum"] != d2["checksum"]


# ---------------------------------------------------------------------------
# FMT-05: Round-trip
# ---------------------------------------------------------------------------


def test_round_trip_all_fields() -> None:
    """Routine.from_dict(routine.to_dict()) round-trips all fields."""
    r = Routine(
        name="my-routine",
        description="A test routine",
        version="2.0.0",
        category="web",
        tags=["browser", "test"],
        programs=["chrome"],
        theme="dark",
        start_from="browser",
        resolution=[2560, 1440],
    )
    d = r.to_dict()
    r2 = Routine.from_dict(d)
    d2 = r2.to_dict()

    assert d == d2


def test_graph_round_trip() -> None:
    """Graph round-trip: 2 nodes + 1 edge survives to_dict -> from_dict."""
    g = OCSDGraph()
    n1 = g.add_node("button", "Start")
    n2 = g.add_node("textbox", "Search")
    g.add_edge(n1, n2, action_type="button")

    r = Routine(name="graph-test")
    r.graph = g
    d = r.to_dict()

    r2 = Routine.from_dict(d)
    g1_json = json.dumps(r.graph.to_dict(), sort_keys=True)
    g2_json = json.dumps(r2.graph.to_dict(), sort_keys=True)
    assert g1_json == g2_json


# ---------------------------------------------------------------------------
# FMT-07: Metadata fields
# ---------------------------------------------------------------------------


def test_metadata_defaults() -> None:
    """Default metadata fields have correct values."""
    r = Routine(name="defaults")
    d = r.to_dict()

    assert d["name"] == "defaults"
    assert d["version"] == "1.0.0"
    # author should be hostname
    import socket
    assert d["author"] == socket.gethostname()
    assert d["category"] == "other"
    assert d["tags"] == []
    assert d["programs"] == []
    # platform mapped from sys.platform
    platform_map = {"win32": "windows", "linux": "ubuntu", "darwin": "macos"}
    assert d["platform"] == platform_map.get(sys.platform, sys.platform)
    assert d["theme"] is None
    assert d["start_from"] == "desktop"
    assert d["resolution"] == [1920, 1080]


def test_valid_categories() -> None:
    """VALID_CATEGORIES contains the required categories."""
    required = {"productivity", "web", "system", "finance", "creative", "other"}
    assert required.issubset(set(VALID_CATEGORIES))


# ---------------------------------------------------------------------------
# Save / Load
# ---------------------------------------------------------------------------


def test_save_creates_routine_json(tmp_path: Path) -> None:
    """save() creates routine.json file with correct content."""
    r = Routine(name="save-test")
    r.save(tmp_path)

    routine_file = tmp_path / "routine.json"
    assert routine_file.exists()

    loaded = json.loads(routine_file.read_text(encoding="utf-8"))
    assert loaded["$schema"] == "ocsd-routine-v1"
    assert loaded["name"] == "save-test"


def test_save_load_roundtrip(tmp_path: Path) -> None:
    """save() then load() reads back identically."""
    r = Routine(name="roundtrip", description="test save/load", tags=["demo"])
    r.save(tmp_path)

    r2 = Routine.load(tmp_path)
    assert r.to_dict() == r2.to_dict()


def test_load_raises_on_checksum_mismatch(tmp_path: Path) -> None:
    """load() raises ValueError on checksum mismatch (tampered file)."""
    r = Routine(name="tamper-test")
    r.save(tmp_path)

    # Tamper with the file
    routine_file = tmp_path / "routine.json"
    data = json.loads(routine_file.read_text(encoding="utf-8"))
    data["steps"] = [{"step_index": 0, "tampered": True}]
    routine_file.write_text(json.dumps(data, indent=2), encoding="utf-8")

    with pytest.raises(ValueError, match="checksum mismatch"):
        Routine.load(tmp_path)


def test_load_raises_on_wrong_schema(tmp_path: Path) -> None:
    """load() raises ValueError on wrong $schema."""
    r = Routine(name="schema-test")
    r.save(tmp_path)

    routine_file = tmp_path / "routine.json"
    data = json.loads(routine_file.read_text(encoding="utf-8"))
    data["$schema"] = "ocsd-routine-v0"
    # Recompute checksum so it doesn't fail on checksum first
    data["checksum"] = calculate_routine_checksum(data["steps"], data["graph"])
    routine_file.write_text(json.dumps(data, indent=2), encoding="utf-8")

    with pytest.raises(ValueError, match="schema"):
        Routine.load(tmp_path)
