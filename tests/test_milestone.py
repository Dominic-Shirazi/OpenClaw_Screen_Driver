"""Milestone integration test: Record → Execute → Verify.

Tests the full Stage 1 pipeline by:
1. Programmatically building a skill graph (simulates recording)
2. Exporting/importing it as a skill JSON file
3. Dry-run executing the path
4. Verifying the ReplayLog shows overall_success
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from core.types import ReplayLog
from executor.pathfinder import find_path
from executor.runner import RunnerEventType, run_path
from mapper.export import export_skill, import_skill, load_skill_from_file, save_skill_to_file
from mapper.graph import OCSDGraph


@pytest.fixture
def login_graph() -> OCSDGraph:
    """Builds a graph representing the login form flow."""
    g = OCSDGraph(skill_id="test-login-skill")

    # Node 1: Username textbox
    n_user = g.add_node(
        element_type="textbox",
        label="Username",
        ocr_text="Enter username",
        x_pct=0.5,
        y_pct=0.35,
    )

    # Node 2: Password textbox
    n_pass = g.add_node(
        element_type="textbox",
        label="Password",
        ocr_text="Enter password",
        x_pct=0.5,
        y_pct=0.50,
    )

    # Node 3: Login button
    n_login = g.add_node(
        element_type="button",
        label="Log In",
        ocr_text="Log In",
        x_pct=0.5,
        y_pct=0.65,
    )

    # Node 4: Success message (read_here / destination)
    n_success = g.add_node(
        element_type="read_here",
        label="Login successful!",
        ocr_text="Login successful!",
        x_pct=0.5,
        y_pct=0.75,
        confidence_threshold=0.60,
    )

    # Edges: username → password → login → success
    g.add_edge(n_user, n_pass, action_type="type", action_payload="ocsd")
    g.add_edge(n_pass, n_login, action_type="type", action_payload="test123")
    g.add_edge(n_login, n_success, action_type="click")

    return g


class TestSkillRoundTrip:
    """Tests skill export → save → load → import."""

    def test_export_and_reimport(self, login_graph: OCSDGraph) -> None:
        """Skill survives a full export → file → import cycle."""
        skill_data = export_skill(
            login_graph,
            name="Login Test",
            description="Test login to OCSD test page",
            author="test-suite",
            version="0.1.0",
            target_app="browser",
            target_url="file:///tests/fixtures/login.html",
        )

        assert skill_data["$schema"] == "ocsd-skill-v1"
        assert skill_data["name"] == "Login Test"
        assert len(skill_data["nodes"]) == 4
        assert len(skill_data["edges"]) == 3
        assert skill_data["checksum"]  # Non-empty

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            json.dump(skill_data, f, indent=2)
            tmp_path = Path(f.name)

        try:
            loaded = load_skill_from_file(tmp_path)
            graph2, metadata = import_skill(loaded)

            assert graph2.node_count == 4
            assert graph2.edge_count == 3
            assert metadata["name"] == "Login Test"
            assert metadata["target_url"] == "file:///tests/fixtures/login.html"
        finally:
            tmp_path.unlink(missing_ok=True)


class TestPathfinding:
    """Tests pathfinding through the login skill graph."""

    def test_finds_full_path(self, login_graph: OCSDGraph) -> None:
        """Pathfinder finds the linear path through all 4 nodes."""
        entry = login_graph.get_entry_nodes()
        assert len(entry) == 1

        exits = login_graph.get_exit_nodes()
        assert len(exits) >= 1

        path = find_path(login_graph, entry[0], exits[0])
        assert len(path) == 4  # username → password → login → success

    def test_read_here_detected(self, login_graph: OCSDGraph) -> None:
        """The success message is detected as a read_here node."""
        read_nodes = login_graph.get_read_here_nodes()
        assert len(read_nodes) == 1


class TestDryRunExecution:
    """Tests dry-run execution of the login skill."""

    def test_dry_run_all_steps_succeed(self, login_graph: OCSDGraph) -> None:
        """Dry run through the full path results in overall_success."""
        entry = login_graph.get_entry_nodes()[0]
        exit_node = login_graph.get_exit_nodes()[0]
        path = find_path(login_graph, entry, exit_node)

        events_received: list[tuple[RunnerEventType, dict]] = []

        def collector(event_type: RunnerEventType, data: dict) -> None:
            events_received.append((event_type, data))

        replay_log = run_path(
            path,
            login_graph,
            dry_run=True,
            event_callback=collector,
        )

        # Core assertion: dry run succeeds
        assert replay_log.overall_success is True
        assert len(replay_log.steps) == 4

        # Every step used dry_run locate method
        for step in replay_log.steps:
            assert step.success is True
            assert step.locate_method == "dry_run"
            assert step.located_at is not None

        # Events were emitted
        event_types = [e[0] for e in events_received]
        assert RunnerEventType.STEP_START in event_types
        assert RunnerEventType.ELEMENT_LOCATED in event_types
        assert RunnerEventType.STEP_COMPLETE in event_types
        assert RunnerEventType.PATH_COMPLETE in event_types
        assert RunnerEventType.PATH_FAILED not in event_types

    def test_dry_run_replay_log_serializes(self, login_graph: OCSDGraph) -> None:
        """ReplayLog.to_dict() produces valid JSON."""
        entry = login_graph.get_entry_nodes()[0]
        exit_node = login_graph.get_exit_nodes()[0]
        path = find_path(login_graph, entry, exit_node)

        replay_log = run_path(path, login_graph, dry_run=True)

        log_dict = replay_log.to_dict()
        # Should be JSON-serializable
        json_str = json.dumps(log_dict)
        parsed = json.loads(json_str)

        assert parsed["overall_success"] is True
        assert parsed["skill_id"] == "test-login-skill"
        assert len(parsed["steps"]) == 4
        assert parsed["duration_ms"] >= 0

    def test_dry_run_with_skip_vlm(self, login_graph: OCSDGraph) -> None:
        """Dry run with skip_vlm_validation still succeeds."""
        entry = login_graph.get_entry_nodes()[0]
        exit_node = login_graph.get_exit_nodes()[0]
        path = find_path(login_graph, entry, exit_node)

        replay_log = run_path(
            path,
            login_graph,
            dry_run=True,
            skip_vlm_validation=True,
        )

        assert replay_log.overall_success is True


class TestFullPipeline:
    """End-to-end: build → export → save → load → pathfind → dry-run → verify."""

    def test_complete_pipeline(self, login_graph: OCSDGraph) -> None:
        """The full Stage 1 pipeline works end-to-end."""
        # Step 1: Export the skill
        skill_data = export_skill(
            login_graph,
            name="Login E2E Test",
            description="Full pipeline test",
            author="test-suite",
            version="0.1.0",
            target_app="browser",
        )

        # Step 2: Save to file
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            tmp_path = Path(f.name)
        save_skill_to_file(skill_data, tmp_path)

        try:
            # Step 3: Load from file (verifies checksum)
            loaded = load_skill_from_file(tmp_path)

            # Step 4: Import into graph
            graph, metadata = import_skill(loaded)
            assert graph.node_count == 4
            assert graph.edge_count == 3

            # Step 5: Find path
            entry_id = loaded["entry_node_id"]
            exit_nodes = graph.get_exit_nodes()
            assert exit_nodes
            path = find_path(graph, entry_id, exit_nodes[0])
            assert len(path) == 4

            # Step 6: Dry-run execute
            replay_log = run_path(path, graph, dry_run=True)

            # Step 7: VERIFY
            assert replay_log.overall_success is True
            assert len(replay_log.steps) == 4
            assert all(s.success for s in replay_log.steps)
            assert replay_log.duration_ms >= 0
            assert replay_log.skill_id == login_graph.skill_id

        finally:
            tmp_path.unlink(missing_ok=True)
