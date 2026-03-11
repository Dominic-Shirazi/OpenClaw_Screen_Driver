"""Wave 2 verification tests for vision.py, watcher.py, and graph.py."""

from __future__ import annotations

import time

import pytest

from core.vision import _extract_json, _extract_json_array, _sanitize_element_type
from core.watcher import is_watching, start_watching, stop_watching
from mapper.graph import OCSDGraph


# ---------------------------------------------------------------------------
# vision.py — JSON parsing resilience
# ---------------------------------------------------------------------------

class TestExtractJson:
    def test_clean_json(self):
        result = _extract_json('{"key": "value"}')
        assert result == {"key": "value"}

    def test_markdown_fenced(self):
        raw = '```json\n{"element_type": "button"}\n```'
        result = _extract_json(raw)
        assert result["element_type"] == "button"

    def test_noisy_json(self):
        raw = 'Here is the result: {"success": true, "confidence": 0.9} done.'
        result = _extract_json(raw)
        assert result["success"] is True

    def test_no_json_raises(self):
        with pytest.raises(ValueError):
            _extract_json("no json here at all")

    def test_empty_string_raises(self):
        with pytest.raises(ValueError):
            _extract_json("")


class TestExtractJsonArray:
    def test_clean_array(self):
        arr = _extract_json_array('[{"a": 1}, {"a": 2}]')
        assert len(arr) == 2

    def test_fenced_array(self):
        raw = '```json\n[{"x": 10}]\n```'
        arr = _extract_json_array(raw)
        assert len(arr) == 1
        assert arr[0]["x"] == 10

    def test_single_object_wrapped(self):
        arr = _extract_json_array('{"rect": {"x": 5}}')
        assert len(arr) == 1


class TestSanitizeElementType:
    def test_valid_types_pass_through(self):
        assert _sanitize_element_type("button") == "button"
        assert _sanitize_element_type("textbox") == "textbox"
        assert _sanitize_element_type("button_nav") == "button_nav"

    def test_case_insensitive(self):
        assert _sanitize_element_type("BUTTON") == "button"
        assert _sanitize_element_type("TextBox") == "textbox"

    def test_fuzzy_mapping(self):
        assert _sanitize_element_type("text_box") == "textbox"
        assert _sanitize_element_type("link") == "button_nav"
        assert _sanitize_element_type("checkbox") == "toggle"
        assert _sanitize_element_type("select") == "dropdown"
        assert _sanitize_element_type("toast") == "notification"

    def test_unknown_fallback(self):
        assert _sanitize_element_type("gibberish") == "unknown"
        assert _sanitize_element_type("") == "unknown"


# ---------------------------------------------------------------------------
# watcher.py — thread lifecycle
# ---------------------------------------------------------------------------

class TestWatcher:
    def setup_method(self):
        """Ensure watcher is stopped before each test."""
        if is_watching():
            stop_watching()

    def teardown_method(self):
        """Cleanup after each test."""
        if is_watching():
            stop_watching()

    def test_not_watching_initially(self):
        assert not is_watching()

    def test_start_and_stop(self):
        events = []
        start_watching(events.append, diff_threshold=0.99, poll_ms=5000)
        time.sleep(0.1)
        assert is_watching()
        stop_watching()
        time.sleep(0.1)
        assert not is_watching()

    def test_double_start_raises(self):
        start_watching(lambda e: None, diff_threshold=0.99, poll_ms=5000)
        time.sleep(0.1)
        with pytest.raises(RuntimeError):
            start_watching(lambda e: None)

    def test_double_stop_is_safe(self):
        start_watching(lambda e: None, diff_threshold=0.99, poll_ms=5000)
        time.sleep(0.1)
        stop_watching()
        stop_watching()  # Should not raise


# ---------------------------------------------------------------------------
# graph.py — CRUD, execution tracking, serialization
# ---------------------------------------------------------------------------

class TestOCSDGraph:
    def _make_graph(self) -> tuple[OCSDGraph, str, str, str]:
        """Creates a test graph with 3 nodes and 2 edges."""
        g = OCSDGraph(skill_id="test-skill")
        n1 = g.add_node("textbox", "Username", x_pct=0.3, y_pct=0.4)
        n2 = g.add_node("textbox", "Password", x_pct=0.3, y_pct=0.5)
        n3 = g.add_node("button_nav", "Login", x_pct=0.3, y_pct=0.6)
        g.add_edge(n1, n2, "textbox", action_payload="tab")
        g.add_edge(n2, n3, "textbox", action_payload="tab")
        return g, n1, n2, n3

    def test_add_and_get_node(self):
        g, n1, _, _ = self._make_graph()
        node = g.get_node(n1)
        assert node["label"] == "Username"
        assert node["element_type"] == "textbox"
        assert node["relative_position"]["region_hint"] == "center_left"

    def test_node_count(self):
        g, _, _, _ = self._make_graph()
        assert g.node_count == 3

    def test_edge_count(self):
        g, _, _, _ = self._make_graph()
        assert g.edge_count == 2

    def test_update_node(self):
        g, n1, _, _ = self._make_graph()
        g.update_node(n1, label="Email Field")
        assert g.get_node(n1)["label"] == "Email Field"

    def test_remove_node(self):
        g, n1, _, _ = self._make_graph()
        g.remove_node(n1)
        assert g.node_count == 2
        assert g.edge_count == 1  # n1->n2 edge also removed

    def test_invalid_element_type_raises(self):
        g = OCSDGraph()
        with pytest.raises(ValueError):
            g.add_node("invalid_type", "Bad")

    def test_invalid_layer_raises(self):
        g = OCSDGraph()
        with pytest.raises(ValueError):
            g.add_node("button", "Bad", layer="nonexistent")

    def test_execution_tracking(self):
        g, n1, n2, _ = self._make_graph()
        # Default rate for untried edge
        assert g.success_rate(n1, n2) == 0.5
        # Record some executions
        g.record_execution(n1, n2, success=True)
        g.record_execution(n1, n2, success=True)
        g.record_execution(n1, n2, success=False)
        rate = g.success_rate(n1, n2)
        assert abs(rate - 2 / 3) < 0.01

    def test_serialization_roundtrip(self):
        g, n1, _, _ = self._make_graph()
        data = g.to_dict()
        g2 = OCSDGraph.from_dict(data)
        assert g2.node_count == 3
        assert g2.edge_count == 2
        assert g2.skill_id == "test-skill"
        assert g2.get_node(n1)["label"] == "Username"

    def test_query_by_type(self):
        g, _, _, _ = self._make_graph()
        textboxes = g.get_nodes_by_type("textbox")
        assert len(textboxes) == 2
        navs = g.get_nodes_by_type("button_nav")
        assert len(navs) == 1

    def test_successors_predecessors(self):
        g, n1, n2, n3 = self._make_graph()
        assert n2 in g.get_successors(n1)
        assert n1 in g.get_predecessors(n2)

    def test_entry_exit_nodes(self):
        g, n1, _, n3 = self._make_graph()
        entries = g.get_entry_nodes()
        assert n1 in entries
        exits = g.get_exit_nodes()
        assert n3 in exits

    def test_get_nonexistent_node_raises(self):
        g = OCSDGraph()
        with pytest.raises(KeyError):
            g.get_node("nonexistent")

    def test_add_edge_missing_node_raises(self):
        g = OCSDGraph()
        n1 = g.add_node("button", "A")
        with pytest.raises(KeyError):
            g.add_edge(n1, "nonexistent", "button")
