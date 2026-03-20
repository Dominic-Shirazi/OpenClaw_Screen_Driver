"""Routine management operations: fork, delete, and inspect.

Standalone CRUD operations invoked from CLI/TUI. Fork creates
independent copies (optionally truncated), delete removes routines
permanently with Rich confirmation, and inspect shows step summaries
as Rich tables or raw JSON.
"""

from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.prompt import Confirm
from rich.table import Table

from routine.discovery import get_routine_dir
from routine.format import Routine

logger = logging.getLogger(__name__)


def fork_routine(
    source_dir: Path,
    new_name: str,
    truncate_at: int | None = None,
    base_dir: Path | None = None,
) -> Path:
    """Create an independent copy of a routine directory.

    Copies all files from *source_dir* into a new directory named
    *new_name*, resets version to ``1.0.0``, and optionally truncates
    steps at a given index (removing orphaned assets).

    Args:
        source_dir: Path to the source routine directory.
        new_name: Name for the forked routine (becomes directory name).
        truncate_at: If set, keep only steps ``[0:truncate_at]`` and
            remove orphaned snippet/embedding files.
        base_dir: Parent directory for the new routine. Defaults to
            :func:`~routine.discovery.get_routine_dir`.

    Returns:
        Path to the newly created routine directory.

    Raises:
        ValueError: If *new_name* directory already exists.
    """
    target_dir = (base_dir or get_routine_dir()) / new_name

    if target_dir.exists():
        raise ValueError(
            f"Target routine directory already exists: {target_dir}"
        )

    shutil.copytree(source_dir, target_dir)

    routine = Routine.load(target_dir)
    routine.name = new_name
    routine.version = "1.0.0"
    routine.updated_at = datetime.now(timezone.utc).isoformat()

    if truncate_at is not None:
        # Collect node_ids that will be kept
        kept_steps = routine.steps[:truncate_at]
        kept_node_ids = {step["node_id"] for step in kept_steps}

        # Collect orphaned node_ids
        all_node_ids = {step["node_id"] for step in routine.steps}
        orphaned_ids = all_node_ids - kept_node_ids

        routine.steps = kept_steps

        # Rebuild graph with only kept nodes
        from mapper.graph import OCSDGraph

        new_graph = OCSDGraph()
        for step in kept_steps:
            nid = step["node_id"]
            new_graph._graph.add_node(nid, node_id=nid)
        routine.graph = new_graph

        # Remove orphaned asset files
        _remove_orphaned_assets(target_dir, orphaned_ids)

    routine.save(target_dir)
    logger.info("Forked routine '%s' -> '%s'", source_dir.name, new_name)
    return target_dir


def _remove_orphaned_assets(
    routine_dir: Path,
    orphaned_ids: set[str],
) -> None:
    """Delete snippet and embedding files for orphaned node IDs.

    Args:
        routine_dir: Path to the routine directory.
        orphaned_ids: Set of node_id strings whose assets should be removed.
    """
    for node_id in orphaned_ids:
        snippet = routine_dir / "snippets" / f"{node_id}.png"
        if snippet.exists():
            snippet.unlink()
            logger.debug("Removed orphaned snippet: %s", snippet)

        embedding = routine_dir / "embeddings" / f"{node_id}.npy"
        if embedding.exists():
            embedding.unlink()
            logger.debug("Removed orphaned embedding: %s", embedding)


def delete_routine(
    routine_dir: Path,
    skip_confirm: bool = False,
) -> bool:
    """Delete a routine directory after Rich confirmation.

    Args:
        routine_dir: Path to the routine directory to delete.
        skip_confirm: If ``True``, skip the confirmation prompt.

    Returns:
        ``True`` if the routine was deleted, ``False`` if the user
        declined.
    """
    routine = Routine.load(routine_dir)
    name = routine.name

    if not skip_confirm:
        confirmed = Confirm.ask(
            f'Delete routine "{name}" and all its files? '
            f"This cannot be undone."
        )
        if not confirmed:
            logger.info("Delete cancelled for routine '%s'", name)
            return False

    shutil.rmtree(routine_dir)
    logger.info("Deleted routine '%s' at %s", name, routine_dir)
    return True


def inspect_routine(
    routine_dir: Path,
    as_json: bool = False,
) -> None:
    """Display a routine summary as Rich table or JSON.

    Args:
        routine_dir: Path to the routine directory to inspect.
        as_json: If ``True``, output the full routine as indented JSON
            to stdout. Otherwise render a Rich table.
    """
    routine = Routine.load(routine_dir)

    if as_json:
        print(json.dumps(routine.to_dict(), indent=2, ensure_ascii=False))
        return

    console = Console()

    # Header info
    console.print(
        f"[bold]{routine.name}[/bold] v{routine.version}  "
        f"({len(routine.steps)} steps, {routine.platform}, "
        f"programs={routine.programs})"
    )
    console.print(f"Created: {routine.created_at}")
    console.print()

    # Steps table
    table = Table(show_header=True, header_style="bold")
    table.add_column("#", width=4, justify="right")
    table.add_column("Action", width=14)
    table.add_column("Label", width=30)
    table.add_column("Bbox", width=20)
    table.add_column("Snippet", width=4, justify="center")

    for step in routine.steps:
        idx = str(step.get("step_index", ""))
        action = step.get("action", "")
        label = step.get("label", "")
        if len(label) > 30:
            label = label[:27] + "..."

        bbox = step.get("bbox", {})
        if isinstance(bbox, dict):
            bbox_str = (
                f"({bbox.get('x', 0)},{bbox.get('y', 0)}) "
                f"{bbox.get('w', 0)}x{bbox.get('h', 0)}"
            )
        else:
            bbox_str = str(bbox)

        snippet_path = routine_dir / "snippets" / f"{step.get('node_id', '')}.png"
        snippet_mark = "[green]\u2713[/green]" if snippet_path.exists() else "[red]-[/red]"

        table.add_row(idx, action, label, bbox_str, snippet_mark)

    console.print(table)
