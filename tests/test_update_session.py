"""Tests for UpdateSession guided-replay orchestrator.

Covers: MGMT-01 (update flow) -- step-by-step walkthrough with
keep/delete/edit/fork controls, graph consistency, version bumping,
orphaned asset cleanup, and loop body reference cleaning.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from mapper.graph import OCSDGraph
from routine.format import Routine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_routine_dir(
    tmp_path: Path,
    name: str = "test-routine",
    num_steps: int = 3,
) -> tuple[Path, Routine]:
    """Create a minimal routine directory with steps, snippets, embeddings, and graph.

    Args:
        tmp_path: Pytest tmp_path fixture.
        name: Routine directory name.
        num_steps: Number of click steps to generate.

    Returns:
        Tuple of (routine_dir, loaded Routine).
    """
    routine_dir = tmp_path / name
    routine_dir.mkdir(parents=True, exist_ok=True)

    steps: list[dict[str, Any]] = []
    graph = OCSDGraph()
    node_ids: list[str] = []

    for i in range(num_steps):
        node_id = f"node-{i:03d}"
        node_ids.append(node_id)
        graph._graph.add_node(node_id, node_id=node_id, element_type="button", label=f"Step {i}")
        steps.append({
            "step_index": i,
            "node_id": node_id,
            "element_type": "button",
            "label": f"Step {i}",
            "action": "click",
            "bbox": {"x": 100 + i * 10, "y": 200, "w": 50, "h": 30},
            "snippet_path": f"snippets/{node_id}.png",
            "embedding_path": f"embeddings/{node_id}.npy",
        })

        # Create dummy asset files
        snippets_dir = routine_dir / "snippets"
        snippets_dir.mkdir(exist_ok=True)
        (snippets_dir / f"{node_id}.png").write_bytes(b"fake-png")

        embeddings_dir = routine_dir / "embeddings"
        embeddings_dir.mkdir(exist_ok=True)
        (embeddings_dir / f"{node_id}.npy").write_bytes(b"fake-npy")

    # Add edges between consecutive nodes
    for i in range(len(node_ids) - 1):
        graph._graph.add_edge(
            node_ids[i], node_ids[i + 1],
            action_type="click",
        )

    routine = Routine(
        name=name,
        version="1.0.0",
        steps=steps,
        graph=graph,
    )
    routine.save(routine_dir)

    # Reload to ensure round-trip consistency
    loaded = Routine.load(routine_dir)
    return routine_dir, loaded


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestUpdateSessionInitialState:
    """Verify initial state after construction."""

    def test_initial_state(self, tmp_path: Path) -> None:
        """current_step starts at 0, is_complete is False."""
        from routine.update_session import UpdateSession

        routine_dir, routine = _make_routine_dir(tmp_path)
        session = UpdateSession(routine, routine_dir)

        assert session.current_step == 0
        assert session.is_complete is False
        assert session.step_count == 3


class TestUpdateSessionNavigation:
    """Verify step navigation controls."""

    def test_keep_and_advance(self, tmp_path: Path) -> None:
        """keep_and_advance() moves current_step forward by 1."""
        from routine.update_session import UpdateSession

        routine_dir, routine = _make_routine_dir(tmp_path)
        session = UpdateSession(routine, routine_dir)
        session.keep_and_advance()

        assert session.current_step == 1

    def test_walk_through_all(self, tmp_path: Path) -> None:
        """Keeping all 3 steps results in is_complete == True."""
        from routine.update_session import UpdateSession

        routine_dir, routine = _make_routine_dir(tmp_path)
        session = UpdateSession(routine, routine_dir)

        for _ in range(3):
            session.keep_and_advance()

        assert session.is_complete is True

    def test_get_current_step(self, tmp_path: Path) -> None:
        """get_current_step() returns the step dict at current index."""
        from routine.update_session import UpdateSession

        routine_dir, routine = _make_routine_dir(tmp_path)
        session = UpdateSession(routine, routine_dir)

        step = session.get_current_step()
        assert step is not None
        assert step["node_id"] == "node-000"


class TestUpdateSessionDeletion:
    """Verify step deletion and its side effects."""

    def test_delete_step(self, tmp_path: Path) -> None:
        """Deleting step 1 produces a routine with 2 steps after save."""
        from routine.update_session import UpdateSession

        routine_dir, routine = _make_routine_dir(tmp_path)
        session = UpdateSession(routine, routine_dir)

        session.keep_and_advance()       # keep step 0
        session.delete_current_step()    # delete step 1
        session.keep_and_advance()       # keep step 2

        result_dir = session.save_updated()
        saved = Routine.load(result_dir)

        assert len(saved.steps) == 2
        assert saved.steps[0]["node_id"] == "node-000"
        assert saved.steps[1]["node_id"] == "node-002"

    def test_delete_removes_graph_node(self, tmp_path: Path) -> None:
        """Deleted step's node_id is no longer in the rebuilt graph."""
        from routine.update_session import UpdateSession

        routine_dir, routine = _make_routine_dir(tmp_path)
        session = UpdateSession(routine, routine_dir)

        session.keep_and_advance()       # keep step 0
        session.delete_current_step()    # delete step 1
        session.keep_and_advance()       # keep step 2

        result_dir = session.save_updated()
        saved = Routine.load(result_dir)

        assert "node-001" not in saved.graph.nodes
        assert "node-000" in saved.graph.nodes
        assert "node-002" in saved.graph.nodes

    def test_delete_cleans_orphan_assets(self, tmp_path: Path) -> None:
        """Deleted step's snippet and embedding files are removed from disk."""
        from routine.update_session import UpdateSession

        routine_dir, routine = _make_routine_dir(tmp_path)
        session = UpdateSession(routine, routine_dir)

        session.keep_and_advance()       # keep step 0
        session.delete_current_step()    # delete step 1
        session.keep_and_advance()       # keep step 2

        session.save_updated()

        # node-001 assets should be deleted
        assert not (routine_dir / "snippets" / "node-001.png").exists()
        assert not (routine_dir / "embeddings" / "node-001.npy").exists()

        # kept assets should still exist
        assert (routine_dir / "snippets" / "node-000.png").exists()
        assert (routine_dir / "snippets" / "node-002.png").exists()


class TestUpdateSessionLoopCleanup:
    """Verify loop body reference cleanup on step deletion."""

    def test_loop_body_cleanup(self, tmp_path: Path) -> None:
        """Loop body references to deleted node_ids are removed."""
        from routine.update_session import UpdateSession

        routine_dir = tmp_path / "loop-routine"
        routine_dir.mkdir(parents=True, exist_ok=True)

        # Build steps: 0=click, 1=click (will be deleted), 2=loop referencing both
        graph = OCSDGraph()
        node_ids = ["node-a", "node-b", "node-loop"]
        for nid in node_ids:
            graph._graph.add_node(nid, node_id=nid, element_type="button", label=nid)

        steps = [
            {
                "step_index": 0, "node_id": "node-a",
                "element_type": "button", "label": "A",
                "action": "click",
                "bbox": {"x": 10, "y": 20, "w": 30, "h": 40},
                "snippet_path": "snippets/node-a.png",
                "embedding_path": "embeddings/node-a.npy",
            },
            {
                "step_index": 1, "node_id": "node-b",
                "element_type": "button", "label": "B",
                "action": "click",
                "bbox": {"x": 50, "y": 60, "w": 30, "h": 40},
                "snippet_path": "snippets/node-b.png",
                "embedding_path": "embeddings/node-b.npy",
            },
            {
                "step_index": 2, "node_id": "node-loop",
                "element_type": "button", "label": "Loop",
                "action": "loop",
                "bbox": {"x": 90, "y": 100, "w": 30, "h": 40},
                "snippet_path": "snippets/node-loop.png",
                "embedding_path": "embeddings/node-loop.npy",
                "loop": {
                    "body_step_node_ids": ["node-a", "node-b"],
                    "exit_condition": "n_iterations",
                    "max_iterations": 5,
                },
            },
        ]

        # Create dummy asset files
        for nid in node_ids:
            (routine_dir / "snippets").mkdir(exist_ok=True)
            (routine_dir / "snippets" / f"{nid}.png").write_bytes(b"fake")
            (routine_dir / "embeddings").mkdir(exist_ok=True)
            (routine_dir / "embeddings" / f"{nid}.npy").write_bytes(b"fake")

        routine = Routine(name="loop-test", version="1.0.0", steps=steps, graph=graph)
        routine.save(routine_dir)
        loaded = Routine.load(routine_dir)

        session = UpdateSession(loaded, routine_dir)
        session.keep_and_advance()       # keep node-a
        session.delete_current_step()    # delete node-b
        session.keep_and_advance()       # keep node-loop

        session.save_updated()
        saved = Routine.load(routine_dir)

        # Find the loop step
        loop_step = next(s for s in saved.steps if s["action"] == "loop")
        assert "node-b" not in loop_step["loop"]["body_step_node_ids"]
        assert "node-a" in loop_step["loop"]["body_step_node_ids"]


class TestUpdateSessionVersioning:
    """Verify version bumping on save."""

    def test_save_bumps_version(self, tmp_path: Path) -> None:
        """Deleting a step causes minor version bump from 1.0.0 to 1.1.0."""
        from routine.update_session import UpdateSession

        routine_dir, routine = _make_routine_dir(tmp_path)
        session = UpdateSession(routine, routine_dir)

        session.keep_and_advance()       # keep step 0
        session.delete_current_step()    # delete step 1
        session.keep_and_advance()       # keep step 2

        session.save_updated()
        saved = Routine.load(routine_dir)

        assert saved.version == "1.1.0"


class TestUpdateSessionFork:
    """Verify save-as-new (fork) flow."""

    def test_save_as_new(self, tmp_path: Path) -> None:
        """save_updated(save_as_new=True) creates a new routine directory."""
        from routine.update_session import UpdateSession

        routine_dir, routine = _make_routine_dir(tmp_path)
        session = UpdateSession(routine, routine_dir)

        # Walk through all steps, keep them all
        for _ in range(3):
            session.keep_and_advance()

        result_dir = session.save_updated(
            save_as_new=True,
            new_name="forked",
        )

        assert result_dir != routine_dir
        assert result_dir.exists()
        forked = Routine.load(result_dir)
        assert forked.name == "forked"
        assert len(forked.steps) == 3
