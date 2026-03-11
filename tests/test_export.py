"""Tests for mapper/export.py.

Verifies skill JSON serialization, deserialization, and integrity checksum.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mapper.export import (
    calculate_checksum,
    export_skill,
    import_skill,
    load_skill_from_file,
    save_skill_to_file,
)
from mapper.graph import OCSDGraph


@pytest.fixture
def sample_graph() -> OCSDGraph:
    """Provides a small graph for testing."""
    graph = OCSDGraph(skill_id="test-skill-uuid")

    # Add 2 nodes
    n1 = graph.add_node(
        element_type="button",
        label="Start Button",
        x_pct=0.2,
        y_pct=0.3,
    )
    n2 = graph.add_node(
        element_type="textbox",
        label="Input Field",
        x_pct=0.5,
        y_pct=0.5,
    )

    # Add 1 edge
    graph.add_edge(
        source_node_id=n1,
        target_node_id=n2,
        action_type="button",
        action_payload="Click start",
    )

    return graph


def test_export_import_roundtrip(sample_graph: OCSDGraph):
    """Verifies that a graph can be exported and imported without loss."""
    # Export
    skill_data = export_skill(
        graph=sample_graph,
        name="Test Skill",
        description="A skill for testing",
        author="Tester",
        version="1.0.0",
        target_app="TestApp",
        tags=["test", "unit"],
    )

    assert skill_data["name"] == "Test Skill"
    assert skill_data["skill_id"] == "test-skill-uuid"
    assert len(skill_data["nodes"]) == 2
    assert len(skill_data["edges"]) == 1
    assert "checksum" in skill_data

    # Import
    imported_graph, metadata = import_skill(skill_data)

    assert imported_graph.skill_id == sample_graph.skill_id
    assert imported_graph.node_count == sample_graph.node_count
    assert imported_graph.edge_count == sample_graph.edge_count

    # Verify node data
    for nid in sample_graph.nodes:
        orig = sample_graph.get_node(nid)
        imp = imported_graph.get_node(nid)
        assert orig["label"] == imp["label"]
        assert orig["element_type"] == imp["element_type"]


def test_file_persistence(sample_graph: OCSDGraph, tmp_path: Path):
    """Verifies saving to and loading from a file."""
    skill_path = tmp_path / "test_skill.json"

    skill_data = export_skill(
        graph=sample_graph,
        name="File Test",
        description="Testing file IO",
        author="Tester",
        version="0.1.0",
        target_app="App",
    )

    # Save
    save_skill_to_file(skill_data, skill_path)
    assert skill_path.exists()

    # Load
    loaded_data = load_skill_from_file(skill_path)
    assert loaded_data["checksum"] == skill_data["checksum"]
    assert loaded_data["name"] == "File Test"


def test_checksum_integrity(sample_graph: OCSDGraph, tmp_path: Path):
    """Verifies that tampering with the file fails checksum validation."""
    skill_path = tmp_path / "tamper_test.json"

    skill_data = export_skill(
        graph=sample_graph,
        name="Integrity Test",
        description="Testing checksum",
        author="Tester",
        version="0.1.0",
        target_app="App",
    )

    save_skill_to_file(skill_data, skill_path)

    # Tamper with the file (change a label in nodes)
    with open(skill_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    data["nodes"][0]["label"] = "TAMPERED LABEL"

    with open(skill_path, "w", encoding="utf-8") as f:
        json.dump(data, f)

    # Loading should now fail
    with pytest.raises(ValueError, match="integrity check failed"):
        load_skill_from_file(skill_path)


def test_checksum_stability():
    """Verifies that checksum is stable regardless of key order or whitespace."""
    nodes = [{"node_id": "a", "val": 1}, {"node_id": "b", "val": 2}]
    edges = [{"edge_id": "e1", "src": "a"}]

    c1 = calculate_checksum(nodes, edges)

    # Reorder nodes in input list
    c2 = calculate_checksum([nodes[1], nodes[0]], edges)

    assert c1 == c2

    # Different key order in dictionary shouldn't matter as we use sort_keys=True
    nodes_alt = [{"val": 1, "node_id": "a"}, {"node_id": "b", "val": 2}]
    c3 = calculate_checksum(nodes_alt, edges)

    assert c1 == c3
