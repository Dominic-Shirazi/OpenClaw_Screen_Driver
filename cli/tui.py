"""Rich TUI for OCSD -- loading screen, arrow-key menu, routine browser.

Provides the interactive terminal interface launched by ``ocsd`` without
arguments.  The TUI gates model initialisation before any Qt overlay
operation and offers searchable routine management.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any

import yaml
from rich.columns import Columns
from rich.console import Console
from rich.live import Live
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt
from rich.table import Table
from rich.text import Text

from cli._keys import read_key

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Menu item definitions
# ---------------------------------------------------------------------------

# (label, command, enabled)
MENU_ITEMS: list[tuple[str, str, bool]] = [
    ("Record Routine", "record", True),
    ("Run Routine", "run", True),
    ("Update Routine", "update", True),
    ("Fork Routine", "fork", True),
    ("List Routines", "list", True),
    ("Inspect Routine", "inspect", True),
    ("Hub Browse", "hub", False),       # V2+ greyed out
    ("Voice Record", "voice", False),   # V2+ greyed out
    ("Quit", "quit", True),
]

# Commands that need a routine selection
_ROUTINE_COMMANDS = {"run", "update", "fork", "inspect"}


# ---------------------------------------------------------------------------
# Feature loading
# ---------------------------------------------------------------------------


def _load_features() -> list[dict[str, str]]:
    """Load feature list from features.yml.

    Reads the ``features.yml`` file co-located with this module.  Falls
    back to a hardcoded list if the file is missing or unreadable.

    Returns:
        List of dicts with ``text`` and ``status`` keys.
    """
    features_path = Path(__file__).parent / "features.yml"
    try:
        with open(features_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        features: list[dict[str, str]] = data.get("features", [])
        if features:
            return features
    except (OSError, yaml.YAMLError) as exc:
        logger.warning("Could not load features.yml: %s", exc)

    # Hardcoded fallback
    return [
        {"text": "Record once, replay forever", "status": "shipped"},
        {"text": "Visual element detection", "status": "shipped"},
        {"text": "Voice-recorded routines", "status": "coming_soon"},
    ]


# ---------------------------------------------------------------------------
# Loading screen
# ---------------------------------------------------------------------------


def show_loading_screen() -> None:
    """Display model initialisation progress and feature ticker.

    Shows a Rich progress bar for each model subsystem (OmniParser,
    CLIP, VLM) and then prints a feature ticker with shipped and
    upcoming features.
    """
    console = Console()

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        console=console,
    )

    omni_task = progress.add_task("OmniParser model", total=1)
    clip_task = progress.add_task("CLIP embeddings", total=1)
    vlm_task = progress.add_task("VLM connectivity", total=1)

    with Live(progress, console=console, refresh_per_second=10):
        # OmniParser
        try:
            from detection.omniparser import get_omniparser  # noqa: PLC0415
            get_omniparser()
        except ImportError:
            logger.info("OmniParser not installed -- skipping")
        except Exception as exc:  # noqa: BLE001
            logger.warning("OmniParser load failed: %s", exc)
        progress.advance(omni_task)

        # CLIP
        try:
            from detection.clip_embedder import CLIPEmbedder  # noqa: PLC0415
            CLIPEmbedder()
        except ImportError:
            logger.info("CLIP embedder not installed -- skipping")
        except Exception as exc:  # noqa: BLE001
            logger.warning("CLIP load failed: %s", exc)
        progress.advance(clip_task)

        # VLM
        try:
            from core.config import get_config  # noqa: PLC0415
            cfg = get_config()
            if hasattr(cfg, "vlm"):
                logger.info("VLM config found")
        except ImportError:
            logger.info("Config module not available -- skipping VLM check")
        except Exception as exc:  # noqa: BLE001
            logger.warning("VLM check failed: %s", exc)
        progress.advance(vlm_task)

    # Feature ticker
    features = _load_features()
    console.print()
    for feat in features:
        if feat["status"] == "shipped":
            console.print(f"  [green]\\u2713[/green] {feat['text']}")
        else:
            console.print(f"  [dim]{feat['text']} -- coming soon[/dim]")
    console.print()


# ---------------------------------------------------------------------------
# Model warmup (fire-and-forget background threads)
# ---------------------------------------------------------------------------


def _start_model_warmup() -> None:
    """Spawn daemon threads to pre-load AI models in background.

    Called when the user selects "record" so models are warm by the
    time recording actually starts.  Each thread swallows all errors
    so missing optional models never block the TUI.
    """

    def _warmup_omniparser() -> None:
        try:
            from core.omniparser import OmniParserProvider  # noqa: PLC0415
            provider = OmniParserProvider()
            provider.warmup()
        except Exception:  # noqa: BLE001
            logger.debug("OmniParser warmup skipped (not available)")

    def _warmup_florence() -> None:
        try:
            from core.florence import load_model  # noqa: PLC0415
            logger.debug("Florence-2 warmup starting")
            load_model()
            logger.debug("Florence-2 warmup complete")
        except Exception:  # noqa: BLE001
            logger.debug("Florence-2 warmup skipped (not available)")

    def _warmup_clip() -> None:
        try:
            from core.embeddings import warmup as warmup_clip  # noqa: PLC0415
            warmup_clip()
        except Exception:  # noqa: BLE001
            logger.debug("CLIP warmup skipped (not available)")

    def _warmup_vlm() -> None:
        try:
            from core.vision import warmup_vlm  # noqa: PLC0415
            warmup_vlm()
        except Exception:  # noqa: BLE001
            logger.debug("VLM warmup skipped (not available)")

    for target in (_warmup_omniparser, _warmup_florence, _warmup_clip, _warmup_vlm):
        t = threading.Thread(target=target, daemon=True)
        t.start()

    logger.debug("Model warmup threads launched")


# ---------------------------------------------------------------------------
# Arrow-key menu
# ---------------------------------------------------------------------------


def _show_menu(include_routine_actions: bool = True) -> tuple[str, dict[str, Any]]:
    """Display arrow-key navigable menu and return user selection.

    Args:
        include_routine_actions: Whether to include routine management
            items.  Currently unused (always True).

    Returns:
        Tuple of ``(command, kwargs)`` where *command* is a menu action
        string and *kwargs* contains any follow-up arguments.
    """
    console = Console()
    items = list(MENU_ITEMS)
    selected = 0

    # Find first enabled index
    while selected < len(items) and not items[selected][2]:
        selected += 1

    def _render_menu() -> Text:
        """Build a Rich Text representation of the current menu state."""
        lines: list[str] = []
        for i, (label, _cmd, enabled) in enumerate(items):
            if i == selected:
                if enabled:
                    lines.append(f"[bold cyan]> {label}[/bold cyan]")
                else:
                    lines.append(f"[dim]> {label} [italic]Feature inbound[/italic][/dim]")
            else:
                if enabled:
                    lines.append(f"  {label}")
                else:
                    lines.append(f"  [dim]{label} [italic]Feature inbound[/italic][/dim]")
        return Text.from_markup("\n".join(lines))

    with Live(_render_menu(), console=console, refresh_per_second=10) as live:
        while True:
            key = read_key()

            if key == "up":
                # Move up, skip disabled, wrap
                start = selected
                selected = (selected - 1) % len(items)
                while not items[selected][2] and selected != start:
                    selected = (selected - 1) % len(items)
            elif key == "down":
                start = selected
                selected = (selected + 1) % len(items)
                while not items[selected][2] and selected != start:
                    selected = (selected + 1) % len(items)
            elif key == "enter":
                if items[selected][2]:  # enabled
                    break
            elif key in ("escape", "q"):
                selected = len(items) - 1  # Quit
                break

            live.update(_render_menu())

    command = items[selected][1]
    kwargs: dict[str, Any] = {}

    if command in _ROUTINE_COMMANDS:
        path = _show_routine_browser()
        if path is None:
            return "quit", {}
        kwargs["routine_path"] = path
    elif command == "record":
        _start_model_warmup()  # fire-and-forget background threads
        name = Prompt.ask("Routine name")
        kwargs["routine_name"] = name

    return command, kwargs


# ---------------------------------------------------------------------------
# Routine browser
# ---------------------------------------------------------------------------


def _show_routine_browser() -> Path | None:
    """Display searchable routine list with type-to-filter.

    Lists routines from the default routine directory.  The user can
    type characters to filter, use arrow keys to navigate, and press
    Enter to select or Escape to cancel.

    Returns:
        Path to the selected routine directory, or ``None`` if cancelled.
    """
    from routine.discovery import list_routines  # noqa: PLC0415

    console = Console()
    routines = list_routines()

    if not routines:
        console.print("[yellow]No routines found.[/yellow]")
        return None

    filter_text = ""
    selected = 0

    def _filtered() -> list[Any]:
        """Return routines matching the current filter."""
        if not filter_text:
            return routines
        return [r for r in routines if filter_text.lower() in r.name.lower()]

    def _render_browser() -> Table:
        """Build a Rich Table for the filtered routine list."""
        filtered = _filtered()
        table = Table(title="Routines", border_style="blue")
        table.add_column("#", style="bold", width=4)
        table.add_column("Name", style="white")
        table.add_column("Version", style="dim")

        for i, r in enumerate(filtered):
            style = "bold cyan" if i == selected else ""
            table.add_row(str(i + 1), r.name, r.schema_version, style=style)

        # Filter prompt at bottom
        caption = f"Type to filter: {filter_text}_"
        table.caption = caption
        return table

    with Live(_render_browser(), console=console, refresh_per_second=10) as live:
        while True:
            key = read_key()
            filtered = _filtered()

            if key == "up":
                if filtered:
                    selected = (selected - 1) % len(filtered)
            elif key == "down":
                if filtered:
                    selected = (selected + 1) % len(filtered)
            elif key == "enter":
                if filtered and 0 <= selected < len(filtered):
                    return filtered[selected].path
                return None
            elif key == "escape":
                return None
            elif key == "backspace":
                filter_text = filter_text[:-1]
                selected = 0
            elif len(key) == 1 and key.isprintable():
                filter_text += key
                selected = 0

            live.update(_render_browser())

    return None  # pragma: no cover


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_tui() -> None:
    """Main TUI entry point -- loading screen then interactive menu loop.

    Displays the loading screen, then enters a menu loop where the user
    can select actions.  Each action is dispatched to the appropriate
    handler.  The loop continues until the user selects Quit.

    Qt-launching commands (record, run, update) follow the sequential
    gate pattern: TUI exits completely, terminal minimizes, Qt runs,
    terminal restores, Rich summary printed, loop continues.
    """
    show_loading_screen()
    console = Console()

    while True:
        command, kwargs = _show_menu()

        if command == "quit":
            return

        if command == "record":
            try:
                from cli._minimize import minimize_terminal, restore_terminal  # noqa: PLC0415
                minimize_terminal()
            except ImportError:
                logger.info("Terminal minimize not available")

            try:
                from recorder.record_flow import cmd_record  # noqa: PLC0415
                cmd_record(kwargs.get("routine_name"))
            except ImportError:
                logger.warning("Record flow not available")
            except Exception as exc:  # noqa: BLE001
                logger.error("Record failed: %s", exc)
            finally:
                try:
                    from cli._minimize import restore_terminal  # noqa: PLC0415
                    restore_terminal()
                except ImportError:
                    pass

            console.print("[green]Recording complete.[/green]")

        elif command == "run":
            routine_path = kwargs.get("routine_path")
            if routine_path is None:
                console.print("[yellow]No routine selected.[/yellow]")
                continue

            try:
                from cli._minimize import minimize_terminal, restore_terminal  # noqa: PLC0415
                minimize_terminal()
            except ImportError:
                logger.info("Terminal minimize not available")

            try:
                import sys as _sys  # noqa: PLC0415
                import threading as _threading  # noqa: PLC0415

                from PyQt6.QtWidgets import QApplication  # noqa: PLC0415

                from recorder.overlay.controller import OverlayController  # noqa: PLC0415
                from routine.replay_overlay import ReplayOverlayAdapter  # noqa: PLC0415
                from routine.runner import run_routine  # noqa: PLC0415

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
                            routine_dir=routine_path, callback=adapter,
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
                status = "[green]SUCCESS[/green]" if result.success else "[red]FAILED[/red]"
                console.print(
                    f"Run complete: {status} "
                    f"({result.steps_completed}/{result.total_steps} steps, "
                    f"{result.duration_ms}ms)"
                )
            except ImportError:
                logger.warning("Run routine not available")
            except Exception as exc:  # noqa: BLE001
                logger.error("Run failed: %s", exc)
            finally:
                try:
                    from cli._minimize import restore_terminal  # noqa: PLC0415
                    restore_terminal()
                except ImportError:
                    pass

        elif command == "update":
            routine_path = kwargs.get("routine_path")
            if routine_path is None:
                console.print("[yellow]No routine selected.[/yellow]")
                continue

            try:
                from cli._minimize import minimize_terminal, restore_terminal  # noqa: PLC0415
                minimize_terminal()
            except ImportError:
                logger.info("Terminal minimize not available")

            try:
                from routine.format import Routine  # noqa: PLC0415
                from routine.update_session import UpdateSession  # noqa: PLC0415
                routine = Routine.load(routine_path)
                session = UpdateSession(routine, routine_path)

                console.print(
                    f"[bold]Updating '{routine.name}' "
                    f"({len(routine.steps)} steps, v{routine.version})[/bold]"
                )

                while not session.is_complete:
                    step = session.get_current_step()
                    if step is None:
                        break
                    step_idx = session.current_step
                    total = session.step_count
                    action = step.get("action", "unknown")
                    label = step.get("label", "")
                    console.print(
                        f"  Step {step_idx + 1}/{total}: "
                        f"[cyan]{action}[/cyan] {label}"
                    )
                    choice = Prompt.ask(
                        "  (k)eep / (d)elete / (s)kip-rest",
                        default="k",
                    )
                    if choice.lower().startswith("d"):
                        session.delete_current_step()
                        console.print("  [yellow]Marked for deletion.[/yellow]")
                    elif choice.lower().startswith("s"):
                        while not session.is_complete:
                            session.keep_and_advance()
                    else:
                        session.keep_and_advance()

                saved_path = session.save_updated()
                updated_routine = Routine.load(saved_path)
                console.print(
                    f"[green]Update complete: "
                    f"{len(updated_routine.steps)} steps, "
                    f"v{updated_routine.version}[/green]"
                )
            except ImportError:
                logger.warning("Update session not available")
            except Exception as exc:  # noqa: BLE001
                logger.error("Update failed: %s", exc)
            finally:
                try:
                    from cli._minimize import restore_terminal  # noqa: PLC0415
                    restore_terminal()
                except ImportError:
                    pass

        elif command == "fork":
            routine_path = kwargs.get("routine_path")
            if routine_path is None:
                console.print("[yellow]No routine selected.[/yellow]")
                continue

            try:
                from routine.management import fork_routine  # noqa: PLC0415
                new_name = Prompt.ask("New routine name")
                new_path = fork_routine(source_dir=routine_path, new_name=new_name)
                console.print(f"[green]Forked -> '{new_name}'[/green]")
                console.print(f"[dim]{new_path}[/dim]")
            except ImportError:
                logger.warning("Fork routine not available")
            except Exception as exc:  # noqa: BLE001
                logger.error("Fork failed: %s", exc)

        elif command == "list":
            try:
                from routine.discovery import list_routines  # noqa: PLC0415
                from cli.output import output_routines  # noqa: PLC0415
                routines = list_routines()
                output_routines(routines, as_json=False)
            except ImportError:
                logger.warning("Routine listing not available")

        elif command == "inspect":
            routine_path = kwargs.get("routine_path")
            if routine_path is None:
                console.print("[yellow]No routine selected.[/yellow]")
                continue

            try:
                from routine.management import inspect_routine  # noqa: PLC0415
                inspect_routine(routine_path)
            except ImportError:
                logger.warning("Inspect routine not available")
            except Exception as exc:  # noqa: BLE001
                logger.error("Inspect failed: %s", exc)

        elif command == "delete":
            routine_path = kwargs.get("routine_path")
            if routine_path is None:
                console.print("[yellow]No routine selected.[/yellow]")
                continue

            try:
                from routine.management import delete_routine  # noqa: PLC0415
                deleted = delete_routine(routine_path)
                if deleted:
                    console.print("[green]Routine deleted.[/green]")
            except ImportError:
                logger.warning("Delete routine not available")
            except Exception as exc:  # noqa: BLE001
                logger.error("Delete failed: %s", exc)
