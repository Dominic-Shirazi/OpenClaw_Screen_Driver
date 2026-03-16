"""OCSD — OpenClaw Screen Driver entry point.

Usage:
    python main.py --record                         Record a workflow (sequential edges)
    python main.py --diagram --name chrome_login     Annotate page layout (no edges, for OmniParser)
    python main.py --execute skills/login.json       Execute a saved skill
    python main.py --execute skills/login.json --to "Submit"  Execute to a label
    python main.py --dry-run --execute skills/login.json      Simulate without acting
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
from pathlib import Path

from core.config import get_config, load_config

logger = logging.getLogger("ocsd")


def _setup_logging(verbose: bool = False) -> None:
    """Configures root logger with console handler."""
    level = logging.DEBUG if verbose else logging.INFO
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    logging.basicConfig(level=level, format=fmt)


def _setup_dpi_awareness() -> None:
    """Enables per-monitor DPI awareness on Windows.

    Without this, screen coordinates from PyAutoGUI may be
    incorrect on high-DPI displays.
    """
    if sys.platform != "win32":
        return

    try:
        import ctypes

        # Try SetProcessDpiAwarenessContext (Win10 1703+)
        try:
            ctypes.windll.user32.SetProcessDpiAwarenessContext(
                ctypes.c_void_p(-4)  # DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2
            )
            logger.debug("DPI awareness: per-monitor v2")
            return
        except (AttributeError, OSError):
            pass

        # Fallback to SetProcessDpiAwareness (Win8.1+)
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
            logger.debug("DPI awareness: per-monitor v1")
            return
        except (AttributeError, OSError):
            pass

        # Last resort: SetProcessDPIAware (Vista+)
        ctypes.windll.user32.SetProcessDPIAware()
        logger.debug("DPI awareness: system-level")
    except Exception as e:
        logger.debug("Could not set DPI awareness: %s", e)


def _ensure_dirs() -> None:
    """Creates required directories from config."""
    cfg = get_config()
    for key in ("skills_dir", "snippets_dir", "replay_logs"):
        path = cfg.get("paths", {}).get(key)
        if path:
            Path(path).mkdir(parents=True, exist_ok=True)


def _setup_signal_handler() -> None:
    """Registers Ctrl+C handler for graceful shutdown."""

    def handler(sig: int, frame: object) -> None:
        logger.info("Interrupted (Ctrl+C). Shutting down...")
        try:
            from core.watcher import stop_watching
            stop_watching()
        except Exception:
            pass
        sys.exit(130)

    signal.signal(signal.SIGINT, handler)


def cmd_compose(args: argparse.Namespace) -> int:
    """Interactively draw edges on a diagram skill.

    Loads a skill JSON (typically a diagram with no edges), renders its
    nodes on the overlay, and lets the user click pairs of nodes to
    create edges. Saves the updated skill when done.
    """
    from PyQt6.QtWidgets import QApplication

    from mapper.export import export_skill, import_skill, load_skill_from_file, save_skill_to_file
    from recorder.overlay import OverlayController, OverlayMode

    skill_path = Path(args.compose_file)
    if not skill_path.exists():
        logger.error("Skill file not found: %s", skill_path)
        return 1

    logger.info("Loading skill for compose: %s", skill_path)
    skill_data = load_skill_from_file(skill_path)
    graph, metadata = import_skill(skill_data)

    logger.info(
        "Compose: '%s' — %d nodes, %d edges",
        metadata.get("name", "?"), graph.node_count, graph.edge_count,
    )

    # Build candidates from graph nodes for overlay rendering
    import pyautogui
    screen_w, screen_h = pyautogui.size()

    candidates: list[dict] = []
    node_id_list = graph.nodes
    for nid in node_id_list:
        nd = graph.get_node(nid)
        pos = nd.get("relative_position", {})
        x_pct = pos.get("x_pct", 0.5)
        y_pct = pos.get("y_pct", 0.5)
        w_pct = pos.get("w_pct", 0.0)
        h_pct = pos.get("h_pct", 0.0)

        x = int(x_pct * screen_w)
        y = int(y_pct * screen_h)
        w = int(w_pct * screen_w) if w_pct > 0 else 60
        h = int(h_pct * screen_h) if h_pct > 0 else 30

        candidates.append({
            "rect": {"x": x - w // 2, "y": y - h // 2, "w": w, "h": h},
            "type_guess": nd.get("element_type", "unknown"),
            "label_guess": nd.get("label", nid[:8]),
            "confidence": 1.0,
            "node_id": nid,
        })

    # Edge-drawing state
    edge_source: list[str | None] = [None]  # mutable container for closure
    edges_added: list[tuple[str, str]] = []

    app = QApplication.instance() or QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)

    def on_element_clicked(
        x: int, y: int, w: int, h: int, candidate: dict | None
    ) -> bool:
        if candidate is None or "node_id" not in candidate:
            logger.info("Compose: click at (%d, %d) — no node matched", x, y)
            return False

        nid = candidate["node_id"]
        label = candidate.get("label_guess", nid[:8])

        if edge_source[0] is None:
            edge_source[0] = nid
            logger.info("Compose: source selected — %s", label)
            return False
        else:
            src = edge_source[0]
            if src == nid:
                logger.info("Compose: same node clicked, ignoring")
                edge_source[0] = None
                return False

            src_label = graph.get_node(src).get("label", src[:8])
            logger.info("Compose: edge %s → %s", src_label, label)

            try:
                graph.add_edge(src, nid, action_type="button")
                edges_added.append((src, nid))
            except Exception as e:
                logger.error("Compose: could not add edge: %s", e)

            edge_source[0] = None
            return True

    def on_mode_changed(mode: OverlayMode) -> None:
        logger.info("Compose overlay mode: %s", mode.name)

    def on_close() -> None:
        logger.info(
            "Compose ended. %d edges added (total: %d edges).",
            len(edges_added), graph.edge_count,
        )
        if edges_added:
            skill_name = metadata.get("name", "composed")
            updated = export_skill(
                graph,
                name=skill_name,
                description=metadata.get("description", f"Composed: {skill_name}"),
                author=metadata.get("author", "ocsd-compose"),
                version=metadata.get("version", "0.1.0"),
                target_app=metadata.get("target_app", "unknown"),
            )
            updated["recording_type"] = "workflow"
            save_skill_to_file(updated, skill_path)
            logger.info("Composed skill saved to %s", skill_path)
        app.quit()

    overlay = OverlayController(
        on_element_clicked=on_element_clicked,
        on_mode_changed=on_mode_changed,
        on_close=on_close,
    )
    overlay.show(start_mode=OverlayMode.RECORD)
    overlay.set_candidates(candidates)

    return app.exec()


def build_parser() -> argparse.ArgumentParser:
    """Builds the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="ocsd",
        description="OpenClaw Screen Driver — AI-powered screen automation",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Debug logging")
    parser.add_argument("--dry-run", action="store_true", dest="dry_run", help="Simulate without acting")
    parser.add_argument("--skip-vlm", action="store_true", dest="skip_vlm", help="Skip VLM validation")

    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("--record", action="store_true", help="Launch recording overlay (workflow mode — sequential edges)")
    group.add_argument("--diagram", action="store_true", help="Launch recording overlay (diagram mode — annotate page, no edges)")
    group.add_argument("--execute", metavar="SKILL_FILE", dest="skill_file", help="Execute a skill JSON")
    group.add_argument("--compose", metavar="SKILL_FILE", dest="compose_file", help="Interactively draw edges on a diagram skill")

    parser.add_argument("--to", metavar="LABEL", dest="target_label", default=None, help="Execute to a specific label")
    parser.add_argument("--name", metavar="NAME", dest="skill_name", default=None, help="Name for recorded skill")

    # Bbox refinement mode (for --record / --diagram)
    refine_group = parser.add_mutually_exclusive_group()
    refine_group.add_argument(
        "--auto-refine", action="store_const", const="auto",
        dest="refine_mode", help="Silently tighten bboxes via OmniParser (default)",
    )
    refine_group.add_argument(
        "--review-refine", action="store_const", const="review",
        dest="refine_mode", help="Show side-by-side comparison for each bbox",
    )
    refine_group.add_argument(
        "--no-refine", action="store_const", const="skip",
        dest="refine_mode", help="Skip bbox refinement entirely",
    )
    parser.set_defaults(refine_mode="auto")

    # Step-through execution mode
    parser.add_argument(
        "--step", action="store_true", dest="step_mode",
        help="Pause before each step during replay (confirm/adjust/skip)",
    )
    parser.add_argument(
        "--debug-ai", action="store_true", dest="debug_ai",
        help="Show locate method, confidence, and reasoning for each step",
    )

    return parser


def _has_mode_arg(args: argparse.Namespace) -> bool:
    """Returns True if any mode flag was explicitly provided."""
    return bool(
        getattr(args, "record", False)
        or getattr(args, "diagram", False)
        or getattr(args, "skill_file", None)
        or getattr(args, "compose_file", None)
    )


def _run_tui(args: argparse.Namespace) -> int:
    """Launches the TUI menu and dispatches the chosen command."""
    try:
        from recorder.tui import launch_menu
    except ImportError:
        logger.debug("rich not installed — falling back to --help")
        build_parser().print_help()
        return 1

    from mapper.execute_controller import cmd_execute
    from recorder.record_controller import cmd_record

    command, kwargs = launch_menu()

    if command == "record":
        args.record = True
        args.diagram = False
        args.skill_name = kwargs.get("skill_name")
        return cmd_record(args)
    elif command == "diagram":
        args.record = False
        args.diagram = True
        args.skill_name = kwargs.get("skill_name")
        return cmd_record(args)
    elif command == "execute":
        args.skill_file = kwargs.get("skill_file", "")
        args.target_label = getattr(args, "target_label", None)
        args.skip_vlm = getattr(args, "skip_vlm", False)
        return cmd_execute(args)
    elif command == "compose":
        args.compose_file = kwargs.get("skill_file", "")
        return cmd_compose(args)
    elif command == "help":
        build_parser().print_help()
        return 0
    else:
        logger.error("Unknown TUI command: %s", command)
        return 1


def main() -> int:
    """Main entry point."""
    parser = build_parser()
    args = parser.parse_args()

    _setup_logging(verbose=args.verbose)
    _setup_signal_handler()
    _setup_dpi_awareness()

    load_config()
    _ensure_dirs()

    logger.info("OCSD v%s", get_config()["ocsd"]["version"])

    if args.record or args.diagram:
        from recorder.record_controller import cmd_record
        return cmd_record(args)
    elif args.skill_file:
        from mapper.execute_controller import cmd_execute
        return cmd_execute(args)
    elif args.compose_file:
        return cmd_compose(args)
    elif not _has_mode_arg(args):
        return _run_tui(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
