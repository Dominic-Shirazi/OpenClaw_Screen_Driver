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

from cli.output import (
    collect_variables,
    output_routines,
    parse_params,
    prepare_run,
    resolve_routine_path,
    show_error,
)

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
def main_callback(
    ctx: typer.Context,
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging"),
) -> None:
    """Launch the TUI when no subcommand is given."""
    if verbose:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )
    else:
        logging.basicConfig(
            level=logging.WARNING,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )

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
        from cli._minimize import minimize_terminal, restore_terminal

        minimize_terminal()
    except ImportError:
        logger.debug("Terminal minimize not available")

    try:
        from recorder.record_flow import cmd_record

        cmd_record(name)
    except Exception as exc:
        show_error("Record Failed", str(exc), fix="Check that all dependencies are installed.")
        raise typer.Exit(code=1) from exc
    finally:
        try:
            from cli._minimize import restore_terminal

            restore_terminal()
        except ImportError:
            pass

    console = Console()
    console.print(Panel("[green]Recording complete.[/green]", title="Record"))


@app.command("run")
def run_command(
    name: str = typer.Argument(..., help="Routine name or path"),
    speed: float = typer.Option(1.0, "--speed", help="Replay speed multiplier"),
    json_output: bool = typer.Option(False, "--json", help="Output result as JSON"),
    param: Optional[list[str]] = typer.Option(None, "--param", help="Parameters as key=value"),
) -> None:
    """Run (replay) a recorded routine."""
    if speed <= 0:
        show_error("Invalid Speed", f"--speed must be greater than 0 (got {speed})")
        raise typer.Exit(code=1)

    try:
        path = resolve_routine_path(name)
    except FileNotFoundError as exc:
        show_error("Routine Not Found", str(exc))
        raise typer.Exit(code=1) from exc

    params = parse_params(param or [])

    # Wire --speed to runner via a shallow copy of execution config
    import copy as _copy

    from core.config import get_config

    cfg = get_config()
    original_execution = cfg.get("execution", {})
    cfg["execution"] = _copy.copy(original_execution)
    cfg["execution"]["human_delay"] = original_execution.get("human_delay", 1.0) / speed

    # Collect variables and prepare temp copy if needed
    all_params = collect_variables(path, params)
    temp_dir: Path | None = None
    run_path = path
    if all_params:
        temp_dir = prepare_run(path, all_params)
        run_path = temp_dir

    try:
        from cli.tui import show_loading_screen

        show_loading_screen()
    except ImportError:
        logger.debug("TUI loading screen not available")

    try:
        import sys as _sys
        import threading as _threading

        from PyQt6.QtWidgets import QApplication

        from recorder.overlay.controller import OverlayController
        from routine.replay_overlay import ReplayOverlayAdapter
        from routine.runner import run_routine

        qt_app = QApplication.instance() or QApplication(_sys.argv)
        qt_app.setQuitOnLastWindowClosed(False)

        controller = OverlayController()
        adapter = ReplayOverlayAdapter(controller)
        controller.show()

        run_result_holder: list = [None]
        run_exc_holder: list = [None]

        def _run_thread() -> None:
            try:
                run_result_holder[0] = run_routine(
                    routine_dir=run_path, callback=adapter,
                )
            except Exception as exc:
                run_exc_holder[0] = exc
            finally:
                qt_app.quit()

        thread = _threading.Thread(target=_run_thread, daemon=True)
        thread.start()
        qt_app.exec()

        if run_exc_holder[0] is not None:
            raise run_exc_holder[0]
        result = run_result_holder[0]
    except Exception as exc:
        cfg["execution"] = original_execution
        show_error("Run Failed", str(exc))
        raise typer.Exit(code=1) from exc
    finally:
        cfg["execution"] = original_execution
        if temp_dir is not None:
            import shutil

            shutil.rmtree(temp_dir.parent, ignore_errors=True)

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

    try:
        from cli._minimize import minimize_terminal, restore_terminal

        minimize_terminal()
    except ImportError:
        logger.debug("Terminal minimize not available")

    from routine.format import Routine
    from routine.update_session import UpdateSession

    try:
        routine = Routine.load(path)
        session = UpdateSession(routine, path)

        console = Console()
        console.print(Panel(
            f"Routine: {routine.name}\n"
            f"Steps: {len(routine.steps)}\n"
            f"Version: {routine.version}",
            title="Update Session",
        ))

        while not session.is_complete:
            step = session.get_current_step()
            if step is None:
                break
            step_idx = session.current_step
            total = session.step_count
            action = step.get("action", "unknown")
            label = step.get("label", "")
            node_id = step.get("node_id", "")
            console.print(Panel(
                f"Action: {action}\n"
                f"Label: {label}\n"
                f"Node ID: {node_id}",
                title=f"Step {step_idx + 1}/{total}",
            ))
            choice = typer.prompt(
                "Action: (k)eep / (d)elete / (s)kip-rest",
                default="k",
            )
            if choice.lower().startswith("d"):
                session.delete_current_step()
                console.print("[yellow]Step marked for deletion.[/yellow]")
            elif choice.lower().startswith("s"):
                # Keep all remaining steps
                while not session.is_complete:
                    session.keep_and_advance()
            else:
                session.keep_and_advance()

        saved_path = session.save_updated()
        updated_routine = Routine.load(saved_path)
        console.print(Panel(
            f"Routine: {updated_routine.name}\n"
            f"Steps: {len(updated_routine.steps)}\n"
            f"Version: {updated_routine.version}",
            title="Update Complete",
        ))
    except Exception as exc:
        show_error("Update Failed", str(exc))
        raise typer.Exit(code=1) from exc
    finally:
        try:
            from cli._minimize import restore_terminal

            restore_terminal()
        except ImportError:
            pass


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


@app.command()
def serve(
    port: int = typer.Option(8420, "--port", help="API server port"),
    host: str = typer.Option("127.0.0.1", "--host", help="Bind address"),
) -> None:
    """Start the OCSD API server for agent/headless use.

    Preloads AI models, then blocks serving HTTP requests.
    Press Ctrl+C for graceful shutdown.
    """
    console = Console()
    console.print("[bold]OCSD API Server[/bold]")
    console.print(f"Binding to {host}:{port}")

    # Preload models
    console.print("[dim]Preloading models...[/dim]")
    try:
        from core.detection import get_detector

        get_detector()
        console.print("[green]  OmniParser loaded[/green]")
    except Exception as exc:
        console.print(f"[yellow]  OmniParser not available: {exc}[/yellow]")

    try:
        from core.embeddings import get_clip_model

        get_clip_model()
        console.print("[green]  CLIP loaded[/green]")
    except Exception as exc:
        console.print(f"[yellow]  CLIP not available: {exc}[/yellow]")

    console.print(f"\n[bold green]OCSD API ready at http://{host}:{port}[/bold green]")
    console.print("[dim]Press Ctrl+C to stop[/dim]\n")

    import uvicorn

    from api.server import app as api_app

    uvicorn.run(api_app, host=host, port=port, log_level="info")


def main() -> None:
    """Entry point for the ``ocsd`` CLI."""
    app()
