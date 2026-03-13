"""TUI launcher for OCSD — rich-based interactive menu.

Provides a power-user terminal interface for selecting OCSD modes
(record, diagram, execute, compose) without memorising CLI flags.
Falls back gracefully when ``rich`` is not installed.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.prompt import IntPrompt, Prompt
    from rich.table import Table
    from rich.text import Text
except ImportError as _exc:
    raise ImportError(
        "The TUI launcher requires the 'rich' package.\n"
        "Install it with:  pip install openclaw-screen-driver[tui]"
    ) from _exc


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MENU_OPTIONS: list[tuple[str, str, str]] = [
    ("1", "Record Workflow", "record"),
    ("2", "Annotate Diagram", "diagram"),
    ("3", "Execute Skill", "execute"),
    ("4", "Connect Steps", "compose"),
    ("5", "Help", "help"),
]

_SKILLS_DIR_DEFAULT = Path("skills")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _scan_skills(skills_dir: Path | None = None) -> list[Path]:
    """Scans the skills directory and returns sorted JSON paths.

    Args:
        skills_dir: Directory to scan.  Defaults to ``skills/``.

    Returns:
        Sorted list of ``*.json`` paths found in *skills_dir*.
    """
    directory = skills_dir or _SKILLS_DIR_DEFAULT
    if not directory.is_dir():
        return []
    return sorted(directory.glob("*.json"))


def _render_banner(console: Console) -> None:
    """Prints the OCSD banner and main menu table.

    Args:
        console: Rich console instance.
    """
    banner = Text.from_markup(
        "[bold cyan]OCSD[/] [dim]// OpenClaw Screen Driver[/]"
    )
    console.print(Panel(banner, border_style="cyan", padding=(0, 2)))

    table = Table(
        show_header=False,
        box=None,
        padding=(0, 2),
        highlight=True,
    )
    table.add_column("key", style="bold green", width=4)
    table.add_column("action", style="white")
    table.add_column("hint", style="dim")

    table.add_row("1", "Record Workflow", "capture sequential steps as a replayable skill")
    table.add_row("2", "Annotate Diagram", "tag UI elements on a page (no edges)")
    table.add_row("3", "Execute Skill", "replay a saved skill JSON")
    table.add_row("4", "Connect Steps", "draw edges between diagram nodes (compose)")
    table.add_row("5", "Help", "CLI reference and shortcuts")

    console.print(table)
    console.print()


def _prompt_skill_name(console: Console) -> str:
    """Asks the user for a skill name with convention hints.

    Args:
        console: Rich console instance.

    Returns:
        A non-empty skill name string.
    """
    console.print(
        "[dim]Skill names use lowercase_with_underscores "
        "(e.g. login_flow, checkout_v2)[/]"
    )
    while True:
        name = Prompt.ask("[bold]Skill name[/]", console=console).strip()
        if name:
            return name
        console.print("[yellow]Name cannot be empty.[/]")


def _prompt_skill_file(
    console: Console,
    skills_dir: Path | None = None,
) -> str:
    """Shows a table of available skill files and lets the user pick one.

    Args:
        console: Rich console instance.
        skills_dir: Override for the skills directory.

    Returns:
        String path to the chosen skill file.
    """
    files = _scan_skills(skills_dir)
    if not files:
        console.print("[yellow]No skill files found.[/]")
        console.print("[dim]Enter the path to a skill JSON manually.[/]")
        while True:
            path = Prompt.ask("[bold]Skill file path[/]", console=console).strip()
            if path:
                return path
            console.print("[yellow]Path cannot be empty.[/]")

    table = Table(title="Available Skills", border_style="blue")
    table.add_column("#", style="bold green", width=4)
    table.add_column("File", style="white")
    table.add_column("Size", style="dim", justify="right")

    for idx, fp in enumerate(files, 1):
        size_kb = fp.stat().st_size / 1024
        table.add_row(str(idx), fp.name, f"{size_kb:.1f} KB")

    console.print(table)
    console.print()

    while True:
        choice = IntPrompt.ask(
            f"[bold]Pick a skill [1-{len(files)}][/]",
            console=console,
        )
        if 1 <= choice <= len(files):
            return str(files[choice - 1])
        console.print(f"[yellow]Enter a number between 1 and {len(files)}.[/]")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def launch_menu(
    skills_dir: Path | None = None,
) -> tuple[str, dict[str, Any]]:
    """Displays the interactive TUI menu and returns the user's selection.

    The menu guides the user through mode selection and any required
    follow-up prompts (skill name, file picker, etc.).

    Args:
        skills_dir: Override for the skills directory used by the file
            browser.  Defaults to ``skills/``.

    Returns:
        A ``(command, kwargs)`` tuple where *command* is one of
        ``"record"``, ``"diagram"``, ``"execute"``, ``"compose"``,
        ``"help"`` and *kwargs* is a dict of relevant arguments for the
        chosen command.
    """
    console = Console()
    _render_banner(console)

    # --- mode selection ---
    valid_keys = {opt[0] for opt in _MENU_OPTIONS}
    while True:
        choice = Prompt.ask(
            "[bold]Select mode[/]",
            choices=list(valid_keys),
            console=console,
        )
        if choice in valid_keys:
            break

    command = next(opt[2] for opt in _MENU_OPTIONS if opt[0] == choice)
    kwargs: dict[str, Any] = {}

    # --- follow-up prompts per command ---
    if command == "record":
        kwargs["skill_name"] = _prompt_skill_name(console)

    elif command == "diagram":
        kwargs["skill_name"] = _prompt_skill_name(console)

    elif command == "execute":
        kwargs["skill_file"] = _prompt_skill_file(console, skills_dir)

    elif command == "compose":
        kwargs["skill_file"] = _prompt_skill_file(console, skills_dir)

    elif command == "help":
        pass  # no extra args needed

    logger.info("TUI selection: %s %s", command, kwargs)
    return command, kwargs
