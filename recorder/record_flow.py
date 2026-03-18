"""Record flow entry point: TUI naming -> overlay -> recording session.

This module provides the new V2 recording entry point that uses the
overlay-integrated pipeline (Phase 1-4) instead of the legacy modal
dialog flow.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def _prompt_routine_name() -> tuple[str, str] | None:
    """Prompt user for routine name and start-from choice via Rich TUI.

    Returns:
        Tuple of (routine_name, start_from) or None if user cancelled.
    """
    try:
        from rich.console import Console
        from rich.prompt import Confirm, Prompt
    except ImportError:
        # Fallback to input() if Rich not available
        name = input("Routine name: ").strip()
        if not name:
            logger.info("Empty routine name, aborting")
            return None
        start_from = (
            input(
                "Does this routine start from the desktop "
                "or an already-open app? [d/a] "
            )
            .strip()
            .lower()
        )
        start_from = "desktop" if start_from != "a" else "app"
        return (name, start_from)

    console = Console()
    name = Prompt.ask("Routine name")
    if not name or not name.strip():
        console.print("[red]Name cannot be empty.[/red]")
        return None
    name = name.strip()

    # Check for existing routine
    routines_dir = Path.home() / ".ocsd" / "routines" / name
    if routines_dir.exists():
        overwrite = Confirm.ask(
            f'Routine "{name}" already exists. Overwrite?',
            default=False,
        )
        if not overwrite:
            return None

    start_from_choice = Prompt.ask(
        "Does this routine start from the desktop or an already-open app?",
        choices=["d", "a"],
        default="d",
    )
    start_from = "desktop" if start_from_choice == "d" else "app"

    return (name, start_from)


def cmd_record(routine_name: str | None = None) -> int:
    """Launch a new recording session.

    If routine_name is not provided, prompts the user via Rich TUI.
    Sets up the overlay, creates a RecordSession, and runs the Qt
    event loop until the session completes (save or abort).

    Args:
        routine_name: Optional pre-set routine name (skip TUI prompt).

    Returns:
        Process exit code (0 = success, 1 = error/abort).
    """
    from PyQt6.QtWidgets import QApplication

    from recorder.overlay.controller import OverlayController
    from recorder.platform_utils import minimize_all_windows
    from recorder.record_session import RecordSession

    # Step 1: Get routine name (TUI prompt or provided)
    start_from = "desktop"
    if routine_name is None:
        result = _prompt_routine_name()
        if result is None:
            return 1
        routine_name, start_from = result

    # Step 2: Create QApplication
    app = QApplication.instance() or QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)

    # Step 3: Minimize windows if starting from desktop
    if start_from == "desktop":
        success = minimize_all_windows()
        if not success and sys.platform == "win32":
            # On Windows, warn but allow continue
            try:
                from rich.console import Console
                from rich.prompt import Confirm

                console = Console()
                console.print(
                    "[yellow]Some windows could not be minimized.[/yellow]"
                )
                if not Confirm.ask("Continue anyway?", default=True):
                    return 1
            except ImportError:
                response = (
                    input(
                        "Some windows could not be minimized. "
                        "Continue anyway? [Y/n] "
                    )
                    .strip()
                    .lower()
                )
                if response == "n":
                    return 1

    logger.info(
        "Starting recording session: %s (start_from=%s)",
        routine_name,
        start_from,
    )

    # Step 4: Create controller and session
    exit_code = 0

    def on_session_complete(success: bool) -> None:
        nonlocal exit_code
        exit_code = 0 if success else 1
        app.quit()

    controller = OverlayController(
        on_save=lambda: None,  # RecordSession will handle
        on_abort=lambda: None,  # RecordSession will handle
    )

    session = RecordSession(
        controller=controller,
        routine_name=routine_name,
        start_from=start_from,
    )

    # Wire session completion to Qt event loop exit
    session.set_on_complete(on_session_complete)

    # Step 5: Pre-load detection models
    try:
        from core.detection import get_detector

        get_detector()
    except (ImportError, RuntimeError, OSError) as e:
        logger.warning("Detection models not available: %s", e)

    # Step 6: Show overlay and start session
    controller.show()
    session.start()

    # Step 7: Run event loop
    app.exec()

    return exit_code
