"""Tests for executor/pathfinder.py — BFS pathfinding over OCSD graph."""

from __future__ import annotations

import pytest

from core.types import PathNotFoundError
from mapper.pathfinder import (
    find_all_paths,
    find_by_label,
    find_path,
    find_read_here_nodes,
)
from mapper.graph import OCSDGraph


def _linear_graph() -> tuple[OCSDGraph, str, str, str]:
    """Creates A -> B -> C linear graph."""
    g = OCSDGraph(skill_id="test")
    a = g.add_node("textbox", "Username", x_pct=0.3, y_pct=0.3)
    b = g.add_node("textbox", "Password", x_pct=0.3, y_pct=0.5)
    c = g.add_node("button_nav", "Login Button", x_pct=0.3, y_pct=0.7)
    g.add_edge(a, b, "textbox", action_payload="tab")
    g.add_edge(b, c, "textbox", action_payload="tab")
    return g, a, b, c


def _diamond_graph() -> tuple[OCSDGraph, str, str, str, str]:
    """Creates a diamond: A -> B, A -> C, B -> D, C -> D."""
    g = OCSDGraph(skill_id="test")
    a = g.add_node("button", "Start")
    b = g.add_node("button", "Path B")
    c = g.add_node("button", "Path C")
    d = g.add_node("button_nav", "End")
    g.add_edge(a, b, "button")
    g.add_edge(a, c, "button")
    g.add_edge(b, d, "button")
    g.add_edge(c, d, "button")
    return g, a, b, c, d


class TestFindPath:
    def test_linear_path(self):
        g, a, b, c = _linear_graph()
        path = find_path(g, a, c)
        assert path == [a, b, c]

    def test_same_node(self):
        g, a, _, _ = _linear_graph()
        path = find_path(g, a, a)
        assert path == [a]

    def test_no_path_raises(self):
        g, a, _, c = _linear_graph()
        # C -> A doesn't exist (graph is directed)
        with pytest.raises(PathNotFoundError):
            find_path(g, c, a)

    def test_missing_node_raises(self):
        g, a, _, _ = _linear_graph()
        with pytest.raises(PathNotFoundError):
            find_path(g, a, "nonexistent")

    def test_prefers_high_success_edges(self):
        g, a, b, c, d = _diamond_graph()
        # Make path A->B->D very successful
        for _ in range(10):
            g.record_execution(a, b, success=True)
            g.record_execution(b, d, success=True)
        # Make path A->C->D unreliable
        for _ in range(10):
            g.record_execution(a, c, success=False)
            g.record_execution(c, d, success=False)
        path = find_path(g, a, d)
        # Should prefer the B path
        assert b in path
        assert c not in path


class TestFindByLabel:
    def test_exact_match(self):
        g, a, _, _ = _linear_graph()
        result = find_by_label(g, "Username")
        assert result == a

    def test_case_insensitive(self):
        g, a, _, _ = _linear_graph()
        result = find_by_label(g, "username")
        assert result == a

    def test_fuzzy_match(self):
        g, _, _, c = _linear_graph()
        result = find_by_label(g, "Login")
        assert result == c

    def test_no_match(self):
        g, _, _, _ = _linear_graph()
        result = find_by_label(g, "zzzzzzzzzzz")
        assert result is None

    def test_empty_label(self):
        g, _, _, _ = _linear_graph()
        result = find_by_label(g, "")
        assert result is None


class TestFindReadHereNodes:
    def test_finds_read_here(self):
        g = OCSDGraph()
        g.add_node("textbox", "Input")
        rh = g.add_node("read_here", "Balance Display")
        result = find_read_here_nodes(g)
        assert rh in result
        assert len(result) == 1

    def test_with_page_hint(self):
        g = OCSDGraph()
        rh1 = g.add_node("read_here", "Balance Display")
        rh2 = g.add_node("read_here", "Order Total")
        result = find_read_here_nodes(g, page_hint="Balance")
        assert rh1 in result
        assert rh2 not in result

    def test_no_read_here(self):
        g, _, _, _ = _linear_graph()
        result = find_read_here_nodes(g)
        assert result == []


class TestFindAllPaths:
    def test_multiple_paths(self):
        g, a, b, c, d = _diamond_graph()
        paths = find_all_paths(g, a, d)
        assert len(paths) == 2

    def test_no_paths(self):
        g, _, _, _, d = _diamond_graph()
        paths = find_all_paths(g, d, "nonexistent")
        assert paths == []

    def test_sorted_by_quality(self):
        g, a, b, c, d = _diamond_graph()
        # Make B path better
        for _ in range(5):
            g.record_execution(a, b, success=True)
            g.record_execution(b, d, success=True)
        for _ in range(5):
            g.record_execution(a, c, success=False)
            g.record_execution(c, d, success=False)
        paths = find_all_paths(g, a, d)
        # Best path (through B) should be first
        assert b in paths[0]
