"""Typer CLI application for OpenClaw Screen Driver.

Provides the ``ocsd`` command-line entry point with subcommands for
recording, running, listing, inspecting, updating, forking, and
deleting routines. Replaces the legacy argparse-based main.py.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel

from cli.output import output_routines, parse_params, resolve_routine_path, show_error

logger = logging.getLogger(__name__)

app = typer.Typer(
    name="ocsd",
    help="OpenClaw Screen Driver -- AI-powered screen automation",
    no_args_is_help=False,
    rich_markup_mode="rich",
    pretty_exceptions_enable=False,
)

# ---------------------------------------------------------------------------
# Hub sub-app (V2 stub)
# ---------------------------------------------------------------------------

hub_app = typer.Typer(name="hub", help="Routine Hub (V2)")


@hub_app.command()
def search(query: str = typer.Argument(..., help="Search query")) -> None:
    """Search the Routine Hub for shared routines (V2 stub)."""
    console = Console()
    console.print(
        "Hub coming soon in V2. For now, routines are stored locally "
        "in ~/.ocsd/routines/"
    )


app.add_typer(hub_app)

# ---------------------------------------------------------------------------
# Main callback (TUI fallback)
# ---------------------------------------------------------------------------


@app.callback(invoke_without_command=True)
def main_callback(ctx: typer.Context) -> None:
    """Launch the TUI when no subcommand is given."""
    if ctx.invoked_subcommand is None:
        from cli.tui import run_tui

        run_tui()


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


@app.command()
def record(
    name: str = typer.Argument(..., help="Routine name"),
) -> None:
    """Record a new routine with the given name."""
    try:
        from cli.tui import show_loading_screen

        show_loading_screen()
    except ImportError:
        logger.debug("TUI loading screen not available")

    try:
        from recorder.record_flow import cmd_record

        cmd_record(name)
    except Exception as exc:
        show_error("Record Failed", str(exc), fix="Check that all dependencies are installed.")
        raise typer.Exit(code=1) from exc


@app.command("run")
def run_command(
    name: str = typer.Argument(..., help="Routine name or path"),
    speed: float = typer.Option(1.0, "--speed", help="Replay speed multiplier"),
    json_output: bool = typer.Option(False, "--json", help="Output result as JSON"),
    param: Optional[list[str]] = typer.Option(None, "--param", help="Parameters as key=value"),
) -> None:
    """Run (replay) a recorded routine."""
    try:
        path = resolve_routine_path(name)
    except FileNotFoundError as exc:
        show_error("Routine Not Found", str(exc))
        raise typer.Exit(code=1) from exc

    params = parse_params(param or [])

    # Wire --speed to runner via config human_delay override
    from core.config import get_config

    cfg = get_config()
    original_delay = cfg.get("execution", {}).get("human_delay", 1.0)
    cfg.setdefault("execution", {})["human_delay"] = 1.0 / speed

    try:
        from cli.tui import show_loading_screen

        show_loading_screen()
    except ImportError:
        logger.debug("TUI loading screen not available")

    try:
        from routine.runner import run_routine

        result = run_routine(routine_dir=path)
    except Exception as exc:
        cfg["execution"]["human_delay"] = original_delay
        show_error("Run Failed", str(exc))
        raise typer.Exit(code=1) from exc
    finally:
        cfg["execution"]["human_delay"] = original_delay

    if json_output:
        print(json.dumps({
            "success": result.success,
            "routine_name": result.routine_name,
            "run_id": result.run_id,
            "steps_completed": result.steps_completed,
            "total_steps": result.total_steps,
            "duration_ms": result.duration_ms,
            "failure_step": result.failure_step,
            "failure_reason": result.failure_reason,
        }, indent=2))
    else:
        console = Console()
        status = "[green]SUCCESS[/green]" if result.success else "[red]FAILED[/red]"
        console.print(Panel(
            f"Routine: {result.routine_name}\n"
            f"Status: {status}\n"
            f"Steps: {result.steps_completed}/{result.total_steps}\n"
            f"Duration: {result.duration_ms}ms\n"
            f"Run ID: {result.run_id}",
            title="Run Result",
        ))


@app.command("list")
def list_command(
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
) -> None:
    """List all discovered routines."""
    from routine.discovery import list_routines

    routines = list_routines()
    output_routines(routines, json_output)


@app.command()
def inspect(
    name: str = typer.Argument(..., help="Routine name or path"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
) -> None:
    """Inspect a routine and show step summary."""
    try:
        path = resolve_routine_path(name)
    except FileNotFoundError as exc:
        show_error("Routine Not Found", str(exc))
        raise typer.Exit(code=1) from exc

    from routine.management import inspect_routine

    inspect_routine(path, as_json=json_output)


@app.command()
def update(
    name: str = typer.Argument(..., help="Routine name or path"),
) -> None:
    """Launch an update session for an existing routine."""
    try:
        path = resolve_routine_path(name)
    except FileNotFoundError as exc:
        show_error("Routine Not Found", str(exc))
        raise typer.Exit(code=1) from exc

    try:
        from cli.tui import show_loading_screen

        show_loading_screen()
    except ImportError:
        logger.debug("TUI loading screen not available")

    from routine.format import Routine
    from routine.update_session import UpdateSession

    routine = Routine.load(path)
    session = UpdateSession(routine, path)

    console = Console()
    console.print(Panel(
        f"Routine: {routine.name}\n"
        f"Steps: {len(routine.steps)}\n"
        f"Version: {routine.version}",
        title="Update Session Started",
    ))


@app.command()
def fork(
    name: str = typer.Argument(..., help="Source routine name or path"),
    new_name: str = typer.Argument(..., help="Name for the forked routine"),
) -> None:
    """Fork (copy) a routine under a new name."""
    try:
        path = resolve_routine_path(name)
    except FileNotFoundError as exc:
        show_error("Routine Not Found", str(exc))
        raise typer.Exit(code=1) from exc

    from routine.management import fork_routine

    try:
        new_path = fork_routine(source_dir=path, new_name=new_name)
        console = Console()
        console.print(f"[green]Forked '{name}' -> '{new_name}'[/green]")
        console.print(f"[dim]{new_path}[/dim]")
    except ValueError as exc:
        show_error("Fork Failed", str(exc))
        raise typer.Exit(code=1) from exc


@app.command()
def delete(
    name: str = typer.Argument(..., help="Routine name or path"),
    yes: bool = typer.Option(False, "--yes", help="Skip confirmation prompt"),
) -> None:
    """Delete a routine permanently."""
    try:
        path = resolve_routine_path(name)
    except FileNotFoundError as exc:
        show_error("Routine Not Found", str(exc))
        raise typer.Exit(code=1) from exc

    from routine.management import delete_routine

    deleted = delete_routine(path, skip_confirm=yes)
    if deleted:
        console = Console()
        console.print(f"[green]Deleted '{name}'[/green]")


def main() -> None:
    """Entry point for the ``ocsd`` CLI."""
    app()
