"""OCSD — OpenClaw Screen Driver entry point.

Usage:
    python main.py --record                         Launch overlay for recording
    python main.py --execute skills/login.json       Execute a saved skill
    python main.py --execute skills/login.json --to "Submit"  Execute to a label
    python main.py --dry-run --execute skills/login.json      Simulate without acting
"""

from __future__ import annotations

import argparse
import json
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

        # Fallback: SetProcessDpiAwareness (Win8.1+)
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
            logger.debug("DPI awareness: per-monitor v1")
            return
        except (AttributeError, OSError):
            pass

        # Last resort: SetProcessDPIAware (Vista+)
        ctypes.windll.user32.SetProcessDPIAware()
        logger.debug("DPI awareness: system-aware (legacy)")
    except Exception as e:
        logger.warning("Could not set DPI awareness: %s", e)


def _ensure_dirs() -> None:
    """Creates output directories from config if they don't exist."""
    cfg = get_config()
    paths = cfg.get("paths", {})
    for key in ("skills_dir", "replay_logs", "snippets_dir"):
        d = paths.get(key)
        if d:
            Path(d).mkdir(parents=True, exist_ok=True)


def cmd_record(args: argparse.Namespace) -> int:
    """Launches the overlay for recording a new skill."""
    from PyQt6.QtWidgets import QApplication

    from recorder.dialog import TagDialog
    from recorder.overlay import OverlayController, OverlayMode

    app = QApplication.instance() or QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)  # Don't quit when TagDialog closes
    logger.info("Starting recording session")

    recorded_elements: list[dict] = []

    def on_element_clicked(x: int, y: int, w: int, h: int, candidate: dict | None) -> bool:
        """Handle an element selection (click or bbox) during recording.

        Returns:
            True if element was recorded, False if skipped/cancelled.
        """
        if w > 0 and h > 0:
            logger.info("Bbox at (%d, %d) %dx%d, candidate=%s", x, y, w, h, candidate is not None)
        else:
            logger.info("Click at (%d, %d), candidate=%s", x, y, candidate is not None)

        # For bounding boxes, position the dialog at the center of the box
        dialog_x = x + w // 2 if w > 0 else x
        dialog_y = y + h // 2 if h > 0 else y

        dialog = TagDialog(
            element_type_guess=candidate.get("type_guess", "unknown") if candidate else "unknown",
            label_guess=candidate.get("label_guess", "") if candidate else "",
            ocr_text=candidate.get("ocr_text") if candidate else None,
            layer_guess=candidate.get("layer_guess", "page_specific") if candidate else "page_specific",
            uia_hint=candidate.get("uia_hint") if candidate else None,
            x=dialog_x,
            y=dialog_y,
        )
        if dialog.exec():
            result = dialog.get_result()
            if result:
                # For bounding boxes, store center as x/y for backward compat
                if w > 0 and h > 0:
                    result["x"] = x + w // 2
                    result["y"] = y + h // 2
                    result["bbox_x"] = x
                    result["bbox_y"] = y
                    result["bbox_w"] = w
                    result["bbox_h"] = h
                else:
                    result["x"] = x
                    result["y"] = y
                    result["bbox_w"] = 0
                    result["bbox_h"] = 0
                recorded_elements.append(result)
                logger.info("Recorded: %s (%s) bbox=%dx%d", result.get("label"), result.get("element_type"), w, h)
                return True
        return False

    def on_mode_changed(mode: OverlayMode) -> None:
        logger.info("Overlay mode: %s", mode.name)

    def on_close() -> None:
        logger.info("Recording ended. Captured %d elements.", len(recorded_elements))
        if recorded_elements:
            _save_recording(recorded_elements, args)
        app.quit()

    overlay = OverlayController(
        on_element_clicked=on_element_clicked,
        on_mode_changed=on_mode_changed,
        on_close=on_close,
    )
    overlay.show()  # starts in PASSTHROUGH — Ctrl+R when ready to record

    return app.exec()


def _save_recording(elements: list[dict], args: argparse.Namespace) -> None:
    """Builds a graph from recorded elements and saves as a skill file."""
    import pyautogui

    from mapper.export import export_skill, save_skill_to_file
    from mapper.graph import OCSDGraph

    screen_w, screen_h = pyautogui.size()
    graph = OCSDGraph()

    prev_node_id: str | None = None
    for elem in elements:
        # Normalize ElementType enum to string value
        raw_et = elem.get("element_type", "unknown")
        et_str = raw_et.value if hasattr(raw_et, "value") else str(raw_et)
        # Compute bounding box percentages (0 for point clicks)
        bbox_w = elem.get("bbox_w", 0)
        bbox_h = elem.get("bbox_h", 0)
        w_pct = bbox_w / screen_w if bbox_w > 0 else 0.0
        h_pct = bbox_h / screen_h if bbox_h > 0 else 0.0

        node_id = graph.add_node(
            element_type=et_str,
            label=elem.get("label", ""),
            ocr_text=elem.get("label", ""),
            x_pct=elem["x"] / screen_w,
            y_pct=elem["y"] / screen_h,
            w_pct=w_pct,
            h_pct=h_pct,
            resolution=(screen_w, screen_h),
        )

        if elem.get("is_destination"):
            graph.update_node(node_id, element_type="read_here")

        if prev_node_id is not None:
            # Map element_type to graph action_type
            # TagDialog returns ElementType enum — normalize to string
            raw_etype = elem.get("element_type", "unknown")
            etype = raw_etype.value if hasattr(raw_etype, "value") else str(raw_etype)
            if etype == "textbox":
                action_type = "textbox"
            elif etype in ("button", "icon", "link", "unknown"):
                action_type = "button"
            elif etype == "button_nav":
                action_type = "button_nav"
            elif etype == "tab":
                action_type = "tab"
            elif etype == "dropdown":
                action_type = "dropdown"
            elif etype == "toggle":
                action_type = "toggle"
            else:
                action_type = "button"
            graph.add_edge(prev_node_id, node_id, action_type=action_type)

        prev_node_id = node_id

    cfg = get_config()
    skills_dir = Path(cfg["paths"]["skills_dir"])
    skill_name = getattr(args, "skill_name", None) or "recording"
    out_path = skills_dir / f"{skill_name}.json"

    skill_data = export_skill(
        graph,
        name=skill_name,
        description=f"Recorded skill: {skill_name}",
        author="ocsd-recorder",
        version="0.1.0",
        target_app="unknown",
    )
    save_skill_to_file(skill_data, out_path)
    logger.info("Skill saved to %s", out_path)


def cmd_execute(args: argparse.Namespace) -> int:
    """Executes a saved skill file."""
    from mapper.export import import_skill, load_skill_from_file
    from mapper.runner import run_skill

    # --skip-vlm overrides vlm_confirm in config at runtime
    if args.skip_vlm:
        cfg = get_config()
        cfg.setdefault("execution", {})["vlm_confirm"] = False
        logger.info("VLM validation disabled (--skip-vlm)")

    skill_path = Path(args.skill_file)
    if not skill_path.exists():
        logger.error("Skill file not found: %s", skill_path)
        return 1

    logger.info("Loading skill: %s", skill_path)
    skill_data = load_skill_from_file(skill_path)
    graph, metadata = import_skill(skill_data)

    logger.info(
        "Skill '%s': %d nodes, %d edges",
        metadata.get("name", "?"),
        graph.node_count,
        graph.edge_count,
    )

    # Determine entry node
    entry_id = skill_data.get("entry_node_id")
    if not entry_id:
        entry_nodes = graph.get_entry_nodes()
        if not entry_nodes:
            logger.error("No entry node found in skill")
            return 1
        entry_id = entry_nodes[0]

    # Determine target node
    if args.target_label:
        # Search nodes by label
        target_id = None
        for nid in graph.nodes:
            node_data = graph.get_node(nid)
            if node_data.get("label", "").lower() == args.target_label.lower():
                target_id = nid
                break
        if not target_id:
            logger.error("No node matching label '%s'", args.target_label)
            return 1
        logger.info("Target: '%s' (node %s)", args.target_label, target_id[:8])
    else:
        exit_nodes = graph.get_exit_nodes()
        if exit_nodes:
            target_id = exit_nodes[0]
        else:
            all_nodes = graph.nodes
            if not all_nodes:
                logger.error("Graph has no nodes")
                return 1
            target_id = all_nodes[-1]

    # Run the skill
    replay_log = run_skill(
        graph,
        start_id=entry_id,
        goal_id=target_id,
        dry_run=args.dry_run,
    )

    # Save replay log
    cfg = get_config()
    logs_dir = Path(cfg["paths"]["replay_logs"])
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f"{replay_log.replay_id}.json"
    with open(log_path, "w") as f:
        json.dump(replay_log.to_dict(), f, indent=2)
    logger.info("Replay log: %s", log_path)

    # Summary
    success_count = sum(1 for s in replay_log.steps if s.success)
    total = len(replay_log.steps)
    status = "SUCCESS" if replay_log.overall_success else "FAILED"
    logger.info(
        "Result: %s (%d/%d steps, %dms)",
        status, success_count, total, replay_log.duration_ms,
    )

    return 0 if replay_log.overall_success else 1


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


def build_parser() -> argparse.ArgumentParser:
    """Builds the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="ocsd",
        description="OpenClaw Screen Driver — AI-powered screen automation",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Debug logging")
    parser.add_argument("--dry-run", action="store_true", dest="dry_run", help="Simulate without acting")
    parser.add_argument("--skip-vlm", action="store_true", dest="skip_vlm", help="Skip VLM validation")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--record", action="store_true", help="Launch recording overlay")
    group.add_argument("--execute", metavar="SKILL_FILE", dest="skill_file", help="Execute a skill JSON")

    parser.add_argument("--to", metavar="LABEL", dest="target_label", default=None, help="Execute to a specific label")
    parser.add_argument("--name", metavar="NAME", dest="skill_name", default=None, help="Name for recorded skill")

    return parser


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

    if args.record:
        return cmd_record(args)
    elif args.skill_file:
        return cmd_execute(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
