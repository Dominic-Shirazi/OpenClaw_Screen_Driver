"""Guided-replay orchestrator for updating existing routines.

UpdateSession walks through a routine step by step, allowing the user
to keep, edit, delete, or fork at each step.  After the walkthrough,
changes are applied atomically: deleted steps are removed, the graph
is rebuilt, loop body references are cleaned, orphaned assets are
deleted, and the version is bumped.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mapper.graph import OCSDGraph
from routine.format import Routine
from routine.management import fork_routine
from routine.version import bump_minor, bump_patch

logger = logging.getLogger(__name__)


class UpdateSession:
    """Guided step-by-step walkthrough for editing an existing routine.

    The session copies the routine's steps to a working list and tracks
    which indices the user has deleted or edited.  Navigation advances
    one step at a time.  After all steps have been visited the caller
    invokes ``save_updated()`` to commit changes.

    Args:
        routine: The Routine instance to update.
        routine_dir: Path to the routine directory on disk.
    """

    def __init__(self, routine: Routine, routine_dir: Path) -> None:
        self._routine = routine
        self._routine_dir = routine_dir
        self._current_step: int = 0
        self._working_steps: list[dict[str, Any]] = list(routine.steps)
        self._deleted_indices: set[int] = set()
        self._edited_indices: set[int] = set()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def current_step(self) -> int:
        """Return the index of the current step in the walkthrough.

        Returns:
            Zero-based step index.
        """
        return self._current_step

    @property
    def is_complete(self) -> bool:
        """Return whether all steps have been visited.

        Returns:
            True when the cursor has advanced past the last step.
        """
        return self._current_step >= len(self._working_steps)

    @property
    def step_count(self) -> int:
        """Return the total number of steps in the working list.

        Returns:
            Number of steps (including those marked for deletion).
        """
        return len(self._working_steps)

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------

    def get_current_step(self) -> dict[str, Any] | None:
        """Return the step dict at the current index.

        Returns:
            Step dictionary, or None if the walkthrough is complete.
        """
        if self.is_complete:
            return None
        return self._working_steps[self._current_step]

    def keep_and_advance(self) -> None:
        """Keep the current step unchanged and advance to the next."""
        self._current_step += 1

    def delete_current_step(self) -> None:
        """Mark the current step for deletion and advance."""
        self._deleted_indices.add(self._current_step)
        self._current_step += 1

    def replace_current_step(self, new_step: dict[str, Any]) -> None:
        """Replace the current step with an edited version and advance.

        Args:
            new_step: Replacement step dictionary.
        """
        self._working_steps[self._current_step] = new_step
        self._edited_indices.add(self._current_step)
        self._current_step += 1

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save_updated(
        self,
        save_as_new: bool = False,
        new_name: str | None = None,
    ) -> Path:
        """Apply all pending changes and persist the routine.

        Steps marked for deletion are removed.  The graph is rebuilt
        from the surviving steps.  Loop body references to deleted
        node_ids are cleaned.  Steps are re-indexed.  The version is
        bumped (minor for structural changes, patch otherwise).
        Orphaned snippet/embedding files are deleted.

        Args:
            save_as_new: If True, fork the routine first then save
                modifications to the fork.
            new_name: Name for the forked routine (required when
                *save_as_new* is True).

        Returns:
            Path to the saved routine directory.
        """
        # 1. Build final step list (exclude deleted indices)
        final_steps = [
            s for i, s in enumerate(self._working_steps)
            if i not in self._deleted_indices
        ]

        # 2. Collect deleted node_ids
        deleted_node_ids = {
            self._working_steps[i]["node_id"]
            for i in self._deleted_indices
        }

        # 3. Clean loop body references
        for step in final_steps:
            if step.get("action") == "loop" and step.get("loop"):
                body_ids = step["loop"].get("body_step_node_ids", [])
                step["loop"]["body_step_node_ids"] = [
                    nid for nid in body_ids if nid not in deleted_node_ids
                ]

        # 4. Re-index steps
        for idx, step in enumerate(final_steps):
            step["step_index"] = idx

        # 5. Rebuild graph from final steps
        new_graph = OCSDGraph()
        remaining_node_ids: list[str] = []
        for step in final_steps:
            nid = step["node_id"]
            new_graph._graph.add_node(
                nid,
                node_id=nid,
                element_type=step.get("element_type", "unknown"),
                label=step.get("label", ""),
            )
            remaining_node_ids.append(nid)

        for i in range(len(remaining_node_ids) - 1):
            src = remaining_node_ids[i]
            tgt = remaining_node_ids[i + 1]
            new_graph._graph.add_edge(
                src, tgt,
                action_type=final_steps[i].get("action", "click"),
            )

        # 6. Determine save target
        has_structural_change = bool(self._deleted_indices or self._edited_indices)

        if save_as_new and new_name:
            target_dir = fork_routine(
                self._routine_dir,
                new_name,
                base_dir=self._routine_dir.parent,
            )
            target_routine = Routine.load(target_dir)
        else:
            target_dir = self._routine_dir
            target_routine = self._routine

        # 7. Apply changes
        target_routine.steps = final_steps
        target_routine.graph = new_graph
        if has_structural_change:
            target_routine.version = bump_minor(target_routine.version)
        else:
            target_routine.version = bump_patch(target_routine.version)
        target_routine.updated_at = datetime.now(timezone.utc).isoformat()
        target_routine.save(target_dir)

        # 8. Remove orphaned assets
        self._clean_orphaned_assets(target_dir, final_steps, deleted_node_ids)

        logger.info(
            "Saved updated routine '%s' v%s (%d steps, %d deleted)",
            target_routine.name,
            target_routine.version,
            len(final_steps),
            len(self._deleted_indices),
        )

        return target_dir

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _clean_orphaned_assets(
        self,
        routine_dir: Path,
        final_steps: list[dict[str, Any]],
        deleted_node_ids: set[str],
    ) -> None:
        """Remove snippet and embedding files that no longer belong to any step.

        Args:
            routine_dir: Path to the routine directory.
            final_steps: List of surviving step dictionaries.
            deleted_node_ids: Set of node_ids that were deleted.
        """
        remaining_ids = {s["node_id"] for s in final_steps}
        snippets_dir = routine_dir / "snippets"
        embeddings_dir = routine_dir / "embeddings"

        for d, ext in [(snippets_dir, ".png"), (embeddings_dir, ".npy")]:
            if not d.exists():
                continue
            for f in d.iterdir():
                if f.suffix == ext and f.stem not in remaining_ids:
                    f.unlink()
                    logger.debug("Removed orphaned asset: %s", f)
