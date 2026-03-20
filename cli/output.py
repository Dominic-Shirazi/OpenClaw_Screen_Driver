"""Shared CLI output helpers for dual JSON/Rich table display.

Provides error panels, routine path resolution, tabular output for
routine listings, parameter parsing utilities, pre-run variable
collection, and temporary routine preparation used across all CLI
subcommands.
"""

from __future__ import annotations

import json
import logging
import shutil
import tempfile
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

from routine.discovery import RoutineInfo, get_routine_dir

logger = logging.getLogger(__name__)


def show_error(title: str, message: str, fix: str | None = None) -> None:
    """Display a red-bordered Rich error panel on stderr.

    Args:
        title: Panel title text.
        message: Main error message body.
        fix: Optional suggestion for how to fix the issue, shown
            in dim text below the message.
    """
    console = Console(stderr=True)
    body = message
    if fix:
        body += f"\n\n[dim]{fix}[/dim]"
    console.print(Panel(body, title=title, border_style="red"))


def resolve_routine_path(name_or_path: str) -> Path:
    """Resolve a routine name or filesystem path to a routine directory.

    Checks whether the argument is an existing file or directory first.
    If it's a file, returns its parent directory. If it's a directory,
    returns it directly. Otherwise looks up the name in the default
    routine storage directory.

    Args:
        name_or_path: Either a routine name (e.g. ``"MyRoutine"``) or
            a filesystem path to a routine directory or routine.json.

    Returns:
        Resolved absolute path to the routine directory.

    Raises:
        FileNotFoundError: If the routine cannot be found.
    """
    candidate = Path(name_or_path)
    if candidate.exists():
        if candidate.is_file():
            return candidate.parent
        return candidate

    routine_dir = get_routine_dir() / name_or_path
    if routine_dir.exists():
        return routine_dir

    raise FileNotFoundError(
        f"Routine '{name_or_path}' not found. "
        f"Checked: {candidate.resolve()}, {routine_dir}"
    )


def output_routines(routines: list[RoutineInfo], as_json: bool) -> None:
    """Display a list of routines as JSON or a Rich table.

    Args:
        routines: List of discovered routines to display.
        as_json: If True, print JSON array to stdout. Otherwise
            render a Rich table with Name, Version, and Path columns.
    """
    if as_json:
        data = [
            {
                "name": r.name,
                "path": str(r.path),
                "version": r.schema_version,
            }
            for r in routines
        ]
        print(json.dumps(data, indent=2))
        return

    console = Console()
    table = Table(show_header=True, header_style="bold")
    table.add_column("Name", style="bold")
    table.add_column("Version", style="dim")
    table.add_column("Path", style="dim")

    for r in routines:
        table.add_row(r.name, r.schema_version, str(r.path))

    console.print(table)


def parse_params(param_list: list[str]) -> dict[str, str]:
    """Parse a list of ``key=value`` strings into a dictionary.

    Args:
        param_list: List of strings in ``"key=value"`` format.

    Returns:
        Dictionary mapping keys to values.

    Raises:
        typer.BadParameter: If any entry does not contain ``=``.
    """
    result: dict[str, str] = {}
    for item in param_list:
        if "=" not in item:
            raise typer.BadParameter(
                f"Parameter '{item}' must be in key=value format"
            )
        key, value = item.split("=", 1)
        result[key] = value
    return result


def collect_variables(
    routine_dir: Path,
    provided_params: dict[str, str],
) -> dict[str, str]:
    """Scan routine steps for variables and prompt for missing ones.

    Loads the routine from *routine_dir*, inspects each step for an
    ``input_spec`` of type ``"variable"``, and interactively prompts
    the user for any variable not already present in *provided_params*.

    Args:
        routine_dir: Path to the routine directory containing routine.json.
        provided_params: Parameters already supplied via ``--param`` flags.

    Returns:
        Merged dictionary of all variable values (provided + prompted).
    """
    from routine.format import Routine  # noqa: PLC0415

    routine = Routine.load(routine_dir)
    merged: dict[str, str] = dict(provided_params)

    for step in routine.steps:
        input_spec = step.get("input_spec", {})
        if input_spec.get("type") != "variable":
            continue

        raw_value: str = input_spec.get("value", "")
        var_name = raw_value.strip("{}")
        if not var_name:
            continue

        if var_name in merged:
            continue

        hint = input_spec.get("hint", f"Enter {var_name}")
        answer = Prompt.ask(f"[bold]{hint}[/]")
        merged[var_name] = answer

    return merged


def prepare_run(
    routine_dir: Path,
    params: dict[str, str],
) -> Path:
    """Create a temporary routine copy with variable values injected.

    Copies *routine_dir* to a temporary directory, loads the routine
    from the copy, replaces ``text_to_type`` fields for variable steps
    with the corresponding values from *params*, and saves the routine
    back to the copy.

    The caller is responsible for cleaning up the returned temporary
    directory (e.g. via ``shutil.rmtree``).

    Args:
        routine_dir: Path to the original routine directory.
        params: Variable name -> value mapping to inject.

    Returns:
        Path to the temporary directory containing the modified routine.
    """
    from routine.format import Routine  # noqa: PLC0415

    temp_dir = Path(tempfile.mkdtemp(prefix="ocsd_run_"))
    # copytree expects the destination not to exist
    temp_routine = temp_dir / "routine"
    shutil.copytree(routine_dir, temp_routine)

    routine = Routine.load(temp_routine)

    for step in routine.steps:
        input_spec = step.get("input_spec", {})
        if input_spec.get("type") != "variable":
            continue
        raw_value: str = input_spec.get("value", "")
        var_name = raw_value.strip("{}")
        if var_name in params:
            step["text_to_type"] = params[var_name]

    routine.save(temp_routine)
    return temp_routine
