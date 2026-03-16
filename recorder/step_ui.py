"""Step-through replay prompt for --step mode.

Provides a TUI prompt between steps during skill replay, letting the
user inspect what's about to happen and decide whether to proceed,
skip, or abort.

When ``rich`` is available, uses styled output.  Falls back to plain
``input()`` otherwise.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def step_through_prompt(
    label: str,
    node_id: str,
    step_num: int,
    action_type: str = "",
    position: tuple[int, int] | None = None,
    confidence: float | None = None,
    method: str = "",
) -> str:
    """Displays a step preview and waits for user input.

    Args:
        label: Human-readable label for the element.
        node_id: The graph node ID.
        step_num: 1-based step number.
        action_type: The action that will be performed (click, type, etc.).
        position: Expected (x, y) screen coordinates, if known.
        confidence: Locate confidence score, if available.
        method: Locate method used (e.g. "yoloe", "clip", "ocr").

    Returns:
        One of: "execute", "skip", "abort".
    """
    try:
        return _rich_prompt(
            label, node_id, step_num, action_type, position, confidence, method,
        )
    except ImportError:
        return _plain_prompt(label, step_num)


def _rich_prompt(
    label: str,
    node_id: str,
    step_num: int,
    action_type: str,
    position: tuple[int, int] | None,
    confidence: float | None,
    method: str,
) -> str:
    """Rich-formatted step prompt."""
    from rich.console import Console
    from rich.panel import Panel
    from rich.prompt import Prompt
    from rich.text import Text

    console = Console()

    # Build info lines
    lines = Text()
    lines.append(f"  Step {step_num}", style="bold cyan")
    lines.append(f"  {label}", style="bold white")
    lines.append(f"  ({node_id[:8]})\n", style="dim")

    if action_type:
        lines.append(f"  Action: ", style="dim")
        lines.append(f"{action_type}\n", style="yellow")

    if position:
        lines.append(f"  Position: ", style="dim")
        lines.append(f"({position[0]}, {position[1]})\n", style="white")

    if confidence is not None:
        style = "green" if confidence >= 0.7 else "yellow" if confidence >= 0.4 else "red"
        lines.append(f"  Confidence: ", style="dim")
        lines.append(f"{confidence:.2f}", style=style)
        if method:
            lines.append(f" via {method}", style="dim")
        lines.append("\n")

    panel = Panel(
        lines,
        title="[bold]Next Step[/bold]",
        border_style="cyan",
        padding=(0, 1),
    )
    console.print(panel)

    choice = Prompt.ask(
        "  [bold][e][/bold]xecute  [bold][s][/bold]kip  [bold][q][/bold]uit",
        choices=["e", "s", "q"],
        default="e",
        show_choices=False,
    )

    return {"e": "execute", "s": "skip", "q": "abort"}.get(choice, "execute")


def _plain_prompt(label: str, step_num: int) -> str:
    """Fallback plain-text prompt when rich is not available."""
    try:
        resp = input(
            f"\n  Step {step_num}: {label}"
            f" — [Enter]=execute, s=skip, q=abort: "
        ).strip().lower()
    except (EOFError, KeyboardInterrupt):
        return "abort"

    if resp in ("q", "quit", "abort"):
        return "abort"
    if resp in ("s", "skip"):
        return "skip"
    return "execute"
