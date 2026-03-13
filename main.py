"""OCSD — OpenClaw Screen Driver entry point.

Usage:
    python main.py --record                         Record a workflow (sequential edges)
    python main.py --diagram --name chrome_login     Annotate page layout (no edges, for YOLO-E)
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
from typing import Any

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


def _try_refine_bbox(
    x: int,
    y: int,
    w: int,
    h: int,
    review_mode: str,
    matched_candidate: dict | None,
) -> tuple[int, int, int, int] | None:
    """Attempts YOLOE bbox refinement for a recorded element.

    For drawn bboxes: calls refine_bbox() to tighten the user's selection.
    For point clicks: infers a bbox from the click point via YOLOE.

    Args:
        x: Element X coordinate.
        y: Element Y coordinate.
        w: Bbox width (0 for point clicks).
        h: Bbox height (0 for point clicks).
        review_mode: One of "auto", "review", "skip".
        matched_candidate: Smart-detect candidate dict (may have 'rect').

    Returns:
        Tuple of (new_x, new_y, new_w, new_h) if refinement accepted,
        or None if rejected/skipped.
    """
    if review_mode == "skip":
        return None

    try:
        from core.capture import screenshot_full
        from core.types import Rect
        from core.yoloe import infer_bbox_at_point, refine_bbox
    except (ImportError, OSError) as e:
        logger.debug("YOLOE not available, skipping bbox refinement: %s", e)
        return None

    try:
        screen = screenshot_full()
    except Exception as e:
        logger.warning("Could not capture screen for refinement: %s", e)
        return None

    yoloe_match = None

    if w > 0 and h > 0:
        # Drawn bbox — refine it
        user_rect = Rect(x=x, y=y, w=w, h=h)
        yoloe_match = refine_bbox(screen, user_rect)
    else:
        # Point click — check if smart_detect already has a bbox
        if matched_candidate and "rect" in matched_candidate:
            r = matched_candidate["rect"]
            if r.get("w", 0) > 0 and r.get("h", 0) > 0:
                logger.info("Using smart_detect bbox for point click")
                return (r["x"], r["y"], r["w"], r["h"])
        # Otherwise ask YOLOE to infer a bbox from the click point
        yoloe_match = infer_bbox_at_point(screen, x, y)

    if yoloe_match is None:
        logger.debug("YOLOE refinement returned no match")
        return None

    refined = yoloe_match.bbox

    if review_mode == "auto":
        logger.info(
            "Auto-refined bbox: (%d,%d) %dx%d → (%d,%d) %dx%d conf=%.2f",
            x, y, w, h, refined.x, refined.y, refined.w, refined.h,
            yoloe_match.confidence,
        )
        return (refined.x, refined.y, refined.w, refined.h)

    # review_mode == "review" — show the RefineDialog
    try:
        from recorder.refine_dialog import RefineDialog

        # Crop original and refined regions for the dialog
        user_crop = screen[y:y + h, x:x + w] if w > 0 and h > 0 else screen[
            max(0, y - 30):y + 30, max(0, x - 30):x + 30
        ]
        yoloe_crop = screen[
            refined.y:refined.y + refined.h,
            refined.x:refined.x + refined.w,
        ]

        if user_crop.size == 0 or yoloe_crop.size == 0:
            return (refined.x, refined.y, refined.w, refined.h)

        dialog = RefineDialog(user_crop, yoloe_crop, refined)
        dialog.exec()
        action, result_rect = dialog.get_result()

        if action == "accepted" and result_rect is not None:
            logger.info("User accepted refined bbox: (%d,%d) %dx%d",
                        result_rect.x, result_rect.y, result_rect.w, result_rect.h)
            return (result_rect.x, result_rect.y, result_rect.w, result_rect.h)
        else:
            logger.info("User rejected refinement, keeping original")
            return None
    except ImportError:
        logger.debug("RefineDialog not available, auto-accepting")
        return (refined.x, refined.y, refined.w, refined.h)
    except Exception as e:
        logger.warning("RefineDialog error: %s, auto-accepting", e)
        return (refined.x, refined.y, refined.w, refined.h)


def _trigger_smart_detect(overlay: Any) -> None:
    """Captures the screen and runs smart detection in the background.

    Results are marshaled back to the Qt main thread and rendered
    as candidate bounding boxes on the overlay.

    Args:
        overlay: The OverlayController to populate with candidates.
    """
    from PyQt6.QtCore import QTimer as _QTimer

    try:
        from core.capture import screenshot_full
        from recorder.smart_detect import detect_ui_elements_async

        screenshot = screenshot_full()
    except Exception as e:
        logger.warning("Smart detect: could not capture screen: %s", e)
        return

    def _on_results(candidates: list[dict]) -> None:
        # Marshal to Qt main thread
        _QTimer.singleShot(0, lambda: overlay.set_candidates(candidates))

    detect_ui_elements_async(screenshot, _on_results)
    logger.info("Smart detection triggered (%d x %d)", screenshot.shape[1], screenshot.shape[0])


def cmd_record(args: argparse.Namespace) -> int:
    """Launches the overlay for recording a new skill.

    Handles both workflow (--record) and diagram (--diagram) modes.
    The overlay is identical — the difference is in how the recording
    is saved (with or without sequential edges).
    """
    from PyQt6.QtWidgets import QApplication

    from recorder.dialog import TagDialog
    from recorder.overlay import OverlayController, OverlayMode

    is_diagram = getattr(args, "diagram", False)
    mode_label = "diagram" if is_diagram else "workflow"
    refine_mode = getattr(args, "refine_mode", "auto")

    app = QApplication.instance() or QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)  # Don't quit when TagDialog closes
    logger.info("Starting %s recording session (refine=%s)", mode_label, refine_mode)

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

        is_bbox = w > 0 and h > 0
        dialog = TagDialog(
            element_type_guess=candidate.get("type_guess", "unknown") if candidate else "unknown",
            label_guess=candidate.get("label_guess", "") if candidate else "",
            ocr_text=candidate.get("ocr_text") if candidate else None,
            layer_guess=candidate.get("layer_guess", "page_specific") if candidate else "page_specific",
            uia_hint=candidate.get("uia_hint") if candidate else None,
            x=dialog_x,
            y=dialog_y,
            is_bbox=is_bbox,
        )
        if dialog.exec():
            result = dialog.get_result()
            if result:
                # Store original coordinates
                cur_x, cur_y, cur_w, cur_h = x, y, w, h
                result["_refinement_status"] = "original"

                # Attempt YOLOE bbox refinement (for all selections)
                refined = _try_refine_bbox(
                    cur_x, cur_y, cur_w, cur_h,
                    refine_mode, candidate,
                )
                if refined is not None:
                    cur_x, cur_y, cur_w, cur_h = refined
                    if w == 0 and h == 0:
                        result["_refinement_status"] = "inferred"
                    elif refine_mode == "auto":
                        result["_refinement_status"] = "auto_refined"
                    else:
                        result["_refinement_status"] = "reviewed"

                # Store final coordinates
                if cur_w > 0 and cur_h > 0:
                    result["x"] = cur_x + cur_w // 2
                    result["y"] = cur_y + cur_h // 2
                    result["bbox_x"] = cur_x
                    result["bbox_y"] = cur_y
                    result["bbox_w"] = cur_w
                    result["bbox_h"] = cur_h
                else:
                    result["x"] = cur_x
                    result["y"] = cur_y
                    result["bbox_w"] = 0
                    result["bbox_h"] = 0

                recorded_elements.append(result)
                logger.info(
                    "Recorded: %s (%s) bbox=%dx%d refine=%s",
                    result.get("label"), result.get("element_type"),
                    cur_w, cur_h, result["_refinement_status"],
                )
                return True
        return False

    def on_mode_changed(mode: OverlayMode) -> None:
        logger.info("Overlay mode: %s", mode.name)
        if mode == OverlayMode.RECORD:
            _trigger_smart_detect(overlay)

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


def _save_snippets_and_embeddings(
    elements: list[dict],
    node_ids: list[str],
    skill_name: str,
    screen_w: int,
    screen_h: int,
) -> None:
    """Saves element crops and CLIP embeddings for replay matching.

    For each element with a bounding box, crops the screen region (with
    30% buffer padding), saves the PNG snippet, and generates a CLIP
    embedding for FAISS similarity search during replay.

    Args:
        elements: List of recorded element dicts with bbox data.
        node_ids: Corresponding graph node IDs (same order as elements).
        skill_name: Skill name for snippet directory.
        screen_w: Screen width in pixels.
        screen_h: Screen height in pixels.
    """
    from core.capture import save_snippet, screenshot_full
    from core.config import get_config

    cfg = get_config()
    crop_buffer = cfg.get("detection", {}).get("crop_buffer_pct", 0.30)

    # Take a single screenshot to crop from (elements were just recorded)
    try:
        full_screen = screenshot_full()
    except Exception as e:
        logger.warning("Could not capture screen for snippets: %s", e)
        return

    embed_count = 0
    for elem, node_id in zip(elements, node_ids):
        bbox_w = elem.get("bbox_w", 0)
        bbox_h = elem.get("bbox_h", 0)

        if bbox_w <= 0 or bbox_h <= 0:
            # Point click — use a small region around the click point
            cx, cy = elem["x"], elem["y"]
            bbox_x = max(0, cx - 30)
            bbox_y = max(0, cy - 30)
            bbox_w = 60
            bbox_h = 60
        else:
            bbox_x = elem.get("bbox_x", elem["x"])
            bbox_y = elem.get("bbox_y", elem["y"])

        # Add buffer padding
        buf_w = int(bbox_w * crop_buffer)
        buf_h = int(bbox_h * crop_buffer)
        x1 = max(0, bbox_x - buf_w)
        y1 = max(0, bbox_y - buf_h)
        x2 = min(screen_w, bbox_x + bbox_w + buf_w)
        y2 = min(screen_h, bbox_y + bbox_h + buf_h)

        # Crop from the full screenshot
        crop = full_screen[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        # Save snippet PNG
        try:
            save_snippet(crop, skill_name, node_id[:12])
        except Exception as e:
            logger.debug("Could not save snippet for %s: %s", node_id[:8], e)

        # Generate CLIP embedding and add to FAISS index
        try:
            import cv2
            from core.embeddings import generate_embedding, save_to_index

            # CLIP expects RGB, our crop is BGR
            rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            embedding = generate_embedding(rgb_crop)
            save_to_index(
                element_id=node_id,
                embedding=embedding,
                x_pct=elem["x"] / screen_w,
                y_pct=elem["y"] / screen_h,
            )
            embed_count += 1
        except ImportError:
            logger.debug("CLIP/FAISS not available, skipping embeddings")
            break  # No point trying for remaining elements
        except Exception as e:
            logger.debug("Could not generate embedding for %s: %s", node_id[:8], e)

    if embed_count > 0:
        logger.info("Saved %d CLIP embeddings to FAISS index", embed_count)


def _save_recording(elements: list[dict], args: argparse.Namespace) -> None:
    """Builds a graph from recorded elements and saves as a skill file.

    In workflow mode (--record), nodes are connected with sequential edges.
    In diagram mode (--diagram), nodes are unordered annotations with no edges
    — used for page layout training (YOLO-E) rather than replay.
    """
    import pyautogui

    from mapper.export import export_skill, save_skill_to_file
    from mapper.graph import OCSDGraph

    is_diagram = getattr(args, "diagram", False)
    screen_w, screen_h = pyautogui.size()
    graph = OCSDGraph()

    # In diagram mode, save a reference screenshot of the page being annotated
    if is_diagram:
        try:
            from core.capture import screenshot_full

            cfg = get_config()
            snippets_dir = Path(cfg["paths"]["snippets_dir"])
            snippets_dir.mkdir(parents=True, exist_ok=True)
            skill_name = getattr(args, "skill_name", None) or "diagram"
            ref_path = snippets_dir / f"{skill_name}_ref.png"
            screenshot_full(str(ref_path))
            logger.info("Diagram reference screenshot: %s", ref_path)
        except Exception as e:
            logger.warning("Could not save reference screenshot: %s", e)

    prev_node_id: str | None = None
    node_ids: list[str] = []
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
        node_ids.append(node_id)

        if elem.get("is_destination"):
            graph.update_node(node_id, element_type="read_here")

        # Workflow mode: connect nodes with sequential edges
        # Diagram mode: no edges — nodes are unordered annotations
        if not is_diagram and prev_node_id is not None:
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

    # Save element snippets and CLIP embeddings for replay matching
    cfg = get_config()
    default_name = "diagram" if is_diagram else "recording"
    skill_name = getattr(args, "skill_name", None) or default_name
    _save_snippets_and_embeddings(elements, node_ids, skill_name, screen_w, screen_h)

    skills_dir = Path(cfg["paths"]["skills_dir"])
    out_path = skills_dir / f"{skill_name}.json"

    recording_type = "diagram" if is_diagram else "workflow"
    skill_data = export_skill(
        graph,
        name=skill_name,
        description=f"Recorded {recording_type}: {skill_name}",
        author="ocsd-recorder",
        version="0.1.0",
        target_app="unknown",
    )
    # Tag the skill data so downstream tools know the recording type
    skill_data["recording_type"] = recording_type
    save_skill_to_file(skill_data, out_path)
    logger.info("%s saved to %s (%d nodes, %d edges)",
                recording_type.capitalize(), out_path,
                graph.node_count, graph.edge_count)


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

    # Build step-through callback if --step or --debug-ai
    step_mode = getattr(args, "step_mode", False)
    debug_ai = getattr(args, "debug_ai", False)
    step_callback = None

    if step_mode or debug_ai:
        from mapper.runner import RunnerEventType

        def _step_event_handler(
            event_type: RunnerEventType, data: dict,
        ) -> None:
            """Event callback for --step and --debug-ai modes."""
            if debug_ai and event_type == RunnerEventType.ELEMENT_LOCATED:
                logger.info(
                    "  [debug-ai] Located via %s at (%s) conf=%.2f",
                    data.get("method", "?"),
                    data.get("point", "?"),
                    data.get("confidence", 0),
                )
            if step_mode and event_type == RunnerEventType.STEP_PREVIEW:
                node_id = data.get("node_id", "?")
                label = data.get("label", node_id[:8] if isinstance(node_id, str) else "?")
                try:
                    from recorder.step_ui import step_through_prompt
                    action = step_through_prompt(
                        label=label,
                        node_id=node_id,
                        step_num=data.get("step", 0) + 1,
                        action_type=data.get("element_type", ""),
                    )
                    if action == "abort":
                        raise KeyboardInterrupt("User aborted via step-through")
                except ImportError:
                    # Fallback to console prompt
                    resp = input(
                        f"\n  Step {data.get('step', 0) + 1}: "
                        f"{label} — [Enter]=execute, s=skip, q=abort: "
                    ).strip().lower()
                    if resp == "q":
                        raise KeyboardInterrupt("User aborted via step-through")

        step_callback = _step_event_handler

    # Run the skill — use orchestrator for full preflight + recovery
    use_orchestrator = not args.skip_vlm  # orchestrator needs VLM for recovery
    if use_orchestrator:
        from mapper.orchestrator import orchestrate_skill

        replay_log = orchestrate_skill(
            graph,
            start_id=entry_id,
            goal_id=target_id,
            dry_run=args.dry_run,
            skip_vlm=args.skip_vlm,
            event_callback=step_callback,
        )
    else:
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


def cmd_compose(args: argparse.Namespace) -> int:
    """Interactively draw edges on a diagram skill.

    Loads a skill JSON (typically a diagram with no edges), renders its
    nodes on the overlay, and lets the user click pairs of nodes to
    create edges. Saves the updated skill when done.

    Workflow:
    1. Load skill JSON and render nodes as candidates on the overlay
    2. User clicks a "source" node, then a "target" node → edge created
    3. Ctrl+Q / ESC saves and exits
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
            # First click — select source
            edge_source[0] = nid
            logger.info("Compose: source selected — %s", label)
            return False  # Don't "record" it, just note it
        else:
            # Second click — create edge
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
            # Save updated skill
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
        dest="refine_mode", help="Silently tighten bboxes via YOLOE (default)",
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
    """Returns True if any mode flag was explicitly provided.

    Args:
        args: Parsed CLI arguments.
    """
    return bool(
        getattr(args, "record", False)
        or getattr(args, "diagram", False)
        or getattr(args, "skill_file", None)
        or getattr(args, "compose_file", None)
    )


def _run_tui(args: argparse.Namespace) -> int:
    """Launches the TUI menu and dispatches the chosen command.

    Falls back to ``parser.print_help()`` when ``rich`` is not installed.

    Args:
        args: Parsed CLI arguments (used for flags like --dry-run).

    Returns:
        Process exit code.
    """
    try:
        from recorder.tui import launch_menu
    except ImportError:
        logger.debug("rich not installed — falling back to --help")
        build_parser().print_help()
        return 1

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
        return cmd_record(args)
    elif args.skill_file:
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
