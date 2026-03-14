"""Recording session controller for OCSD.

Manages the full recording flow: overlay setup, element selection,
smart detection, bbox refinement, VLM labeling, snippet/embedding
saving, and graph export.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

from core.config import get_config

logger = logging.getLogger(__name__)


def _try_refine_bbox(
    x: int,
    y: int,
    w: int,
    h: int,
    review_mode: str,
    matched_candidate: dict | None,
) -> tuple[int, int, int, int] | None:
    """Attempts OmniParser bbox refinement for a recorded element.

    For drawn bboxes: finds the detection with the highest IoU overlap.
    For point clicks: uses matched_candidate rect if available, otherwise
    finds the smallest OmniParser detection containing the click point.

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
        from core.detection import get_detector
        from core.types import Rect
    except (ImportError, OSError) as e:
        logger.debug("Detection module not available, skipping bbox refinement: %s", e)
        return None

    try:
        screen = screenshot_full()
    except Exception as e:
        logger.warning("Could not capture screen for refinement: %s", e)
        return None

    # Detect all elements on screen
    try:
        detector = get_detector()
        candidates = detector.detect(screen)
    except Exception as e:
        logger.debug("Detection failed during refinement: %s", e)
        return None
    finally:
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    if not candidates:
        return None

    # Check if smart_detect already has a bbox for point clicks
    if w == 0 and h == 0:
        if matched_candidate and "rect" in matched_candidate:
            r = matched_candidate["rect"]
            if r.get("w", 0) > 0 and r.get("h", 0) > 0:
                logger.info("Using smart_detect bbox for point click")
                return (r["x"], r["y"], r["w"], r["h"])

    # Find best overlapping detection
    best_overlap = 0.0
    best_rect = None

    for c in candidates:
        r = c["rect"]
        if w > 0 and h > 0:
            # Drawn bbox — compute IoU
            ix1 = max(x, r["x"])
            iy1 = max(y, r["y"])
            ix2 = min(x + w, r["x"] + r["w"])
            iy2 = min(y + h, r["y"] + r["h"])
            if ix2 > ix1 and iy2 > iy1:
                inter = (ix2 - ix1) * (iy2 - iy1)
                union = w * h + r["w"] * r["h"] - inter
                iou = inter / union if union > 0 else 0
                if iou > best_overlap:
                    best_overlap = iou
                    best_rect = r
        else:
            # Point click — find detection containing (x, y)
            if (r["x"] <= x <= r["x"] + r["w"]
                    and r["y"] <= y <= r["y"] + r["h"]):
                area = r["w"] * r["h"]
                # Prefer smallest containing bbox
                if best_rect is None or area < best_rect["w"] * best_rect["h"]:
                    best_overlap = 1.0
                    best_rect = r

    if best_rect is None:
        logger.debug("No detection overlaps the selection")
        return None

    refined = Rect(best_rect["x"], best_rect["y"], best_rect["w"], best_rect["h"])

    if review_mode == "auto":
        logger.info(
            "Auto-refined bbox: (%d,%d) %dx%d → (%d,%d) %dx%d",
            x, y, w, h, refined.x, refined.y, refined.w, refined.h,
        )
        return (refined.x, refined.y, refined.w, refined.h)

    # review_mode == "review" — show the RefineDialog
    try:
        from recorder.refine_dialog import RefineDialog

        # Crop original and refined regions for the dialog
        user_crop = screen[y:y + h, x:x + w] if w > 0 and h > 0 else screen[
            max(0, y - 30):y + 30, max(0, x - 30):x + 30
        ]
        refined_crop = screen[
            refined.y:refined.y + refined.h,
            refined.x:refined.x + refined.w,
        ]

        if user_crop.size == 0 or refined_crop.size == 0:
            return (refined.x, refined.y, refined.w, refined.h)

        dialog = RefineDialog(user_crop, refined_crop, refined)
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


def _auto_snip(x: int, y: int, radius: int = 120) -> dict | None:
    """Captures a region around a click and runs detection to find an element.

    Used when the user clicks on an area with no existing candidate —
    snips the region, runs OmniParser, and returns the best detection
    as a candidate dict with auto-adjusted borders.

    Args:
        x: Click X coordinate on screen.
        y: Click Y coordinate on screen.
        radius: Pixel radius around click to capture.

    Returns:
        Candidate dict with rect/type_guess/label_guess, or None if
        no element was detected.
    """
    try:
        from core.capture import screenshot_full
        from core.detection import get_detector
    except ImportError:
        logger.debug("Detection module not available for auto-snip")
        return None

    try:
        screen = screenshot_full()
    except Exception as e:
        logger.warning("Auto-snip: could not capture screen: %s", e)
        return None

    sh, sw = screen.shape[:2]
    x1 = max(0, x - radius)
    y1 = max(0, y - radius)
    x2 = min(sw, x + radius)
    y2 = min(sh, y + radius)
    crop = screen[y1:y2, x1:x2]

    if crop.size == 0:
        return None

    try:
        detector = get_detector()
        candidates = detector.detect(crop)
    except Exception as e:
        logger.debug("Auto-snip detection failed: %s", e)
        return None
    finally:
        # Free the full screenshot — only crop is needed
        del screen
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    if not candidates:
        return None

    # Find the candidate closest to the click point (in crop coords)
    click_cx = x - x1
    click_cy = y - y1
    best = None
    best_dist = float("inf")

    for c in candidates:
        r = c["rect"]
        cx = r["x"] + r["w"] / 2
        cy = r["y"] + r["h"] / 2
        dist = ((cx - click_cx) ** 2 + (cy - click_cy) ** 2) ** 0.5
        if dist < best_dist:
            best = c
            best_dist = dist

    if best is None:
        return None

    # Offset rect back to screen coordinates
    r = best["rect"]
    best["rect"] = {
        "x": r["x"] + x1,
        "y": r["y"] + y1,
        "w": r["w"],
        "h": r["h"],
    }
    logger.info(
        "Auto-snip found element at (%d,%d) %dx%d near click (%d,%d)",
        best["rect"]["x"], best["rect"]["y"],
        best["rect"]["w"], best["rect"]["h"], x, y,
    )
    return best


def _trigger_smart_detect(
    overlay: Any,
    recorded_elements: list[dict],
    refine_mode: str,
    on_element_clicked: Any,
) -> None:
    """Captures the screen and runs smart detection in the background.

    After detection completes, renders candidates on the overlay and
    starts the one-by-one review flow where each element is presented
    to the user via TagDialog with pre-filled LM data.

    Args:
        overlay: The OverlayController to populate with candidates.
        recorded_elements: List to append accepted elements to.
        refine_mode: Bbox refinement mode ("auto", "review", "skip").
        on_element_clicked: The element recording callback.
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
        def _set_and_review() -> None:
            overlay.set_candidates(candidates)
            if candidates:
                # Start review after a brief delay so the user sees all boxes
                _QTimer.singleShot(
                    1500,
                    lambda: _start_review(
                        overlay, candidates, recorded_elements,
                        refine_mode, on_element_clicked,
                    ),
                )

        # Marshal to Qt main thread
        _QTimer.singleShot(0, _set_and_review)

    detect_ui_elements_async(screenshot, _on_results)
    logger.info("Smart detection triggered (%d x %d)", screenshot.shape[1], screenshot.shape[0])


def _start_review(
    overlay: Any,
    candidates: list[dict],
    recorded_elements: list[dict],
    refine_mode: str,
    on_element_clicked: Any,
) -> None:
    """Iterates through detected candidates one-by-one for review.

    For each candidate, highlights it on the overlay, opens a TagDialog
    with pre-filled LM data (type, label, caption). User can accept
    (Enter/Confirm) or skip. After all reviewed, the user can manually
    add missed elements by clicking/dragging.

    Args:
        overlay: The OverlayController with candidates rendered.
        candidates: List of candidate dicts from detection.
        recorded_elements: List to append accepted elements to.
        refine_mode: Bbox refinement mode.
        on_element_clicked: The element recording callback for manual adds.
    """
    from recorder.dialog import TagDialog

    logger.info("Starting review of %d detected elements...", len(candidates))

    def _review_one(index: int, candidate: dict[str, Any]) -> bool:
        """Review a single candidate. Returns True if accepted."""
        rect = candidate.get("rect", {})
        x = rect.get("x", 0)
        y = rect.get("y", 0)
        w = rect.get("w", 0)
        h = rect.get("h", 0)

        dialog_x = x + w // 2
        dialog_y = y + h // 2

        type_guess = candidate.get("type_guess", "unknown")
        label_guess = candidate.get("label_guess", "")
        florence_caption = candidate.get("florence_caption", "")

        dialog = TagDialog(
            element_type_guess=type_guess,
            label_guess=florence_caption or label_guess,
            ocr_text=candidate.get("ocr_text"),
            layer_guess=candidate.get("layer_guess", "page_specific"),
            uia_hint=candidate.get("uia_hint"),
            x=dialog_x,
            y=dialog_y,
            is_bbox=True,
        )

        if dialog.exec():
            result = dialog.get_result()
            if result:
                result["_refinement_status"] = "detected"
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
                logger.info(
                    "Review accepted [%d/%d]: %s (%s)",
                    index + 1, len(candidates),
                    result.get("label"), result.get("element_type"),
                )
                return True

        logger.info("Review skipped [%d/%d]", index + 1, len(candidates))
        return False

    overlay.start_review(_review_one)
    logger.info(
        "Review complete. %d elements recorded so far. "
        "Click/drag to add missed elements.",
        len(recorded_elements),
    )


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


def _save_recording(
    elements: list[dict],
    args: Any,
) -> None:
    """Builds a graph from recorded elements and saves as a skill file.

    In workflow mode (--record), nodes are connected with sequential edges.
    In diagram mode (--diagram), nodes are unordered annotations with no edges
    — used for page layout training (OmniParser) rather than replay.

    Args:
        elements: List of recorded element dicts.
        args: Parsed CLI arguments (needs skill_name, diagram flag).
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


def cmd_record(args: Any) -> int:
    """Launches the overlay for recording a new skill.

    Handles both workflow (--record) and diagram (--diagram) modes.
    The overlay is identical — the difference is in how the recording
    is saved (with or without sequential edges).

    Args:
        args: Parsed CLI arguments.

    Returns:
        Process exit code.
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

        For point clicks with no matching candidate, auto-snip is triggered:
        captures a region around the click, runs detection, and uses the
        closest detected element with auto-adjusted borders.

        Returns:
            True if element was recorded, False if skipped/cancelled.
        """
        # Auto-snip: if point click with no candidate, try to detect element
        if w == 0 and h == 0 and candidate is None:
            snipped = _auto_snip(x, y)
            if snipped:
                candidate = snipped
                r = snipped["rect"]
                x, w, h = r["x"], r["w"], r["h"]
                y = r["y"]
                logger.info("Auto-snip adjusted click to bbox (%d,%d) %dx%d", x, y, w, h)

        if w > 0 and h > 0:
            logger.info("Bbox at (%d, %d) %dx%d, candidate=%s", x, y, w, h, candidate is not None)
        else:
            logger.info("Click at (%d, %d), candidate=%s", x, y, candidate is not None)

        # For bounding boxes, position the dialog at the center of the box
        dialog_x = x + w // 2 if w > 0 else x
        dialog_y = y + h // 2 if h > 0 else y

        # VLM labeling: crop element + 30% buffer, ask VLM for label
        vlm_label = ""
        vlm_type = ""
        if candidate and candidate.get("rect"):
            try:
                from core.vision import analyze_crop_array
                from core.capture import screenshot_full

                screen = screenshot_full()
                r = candidate["rect"]
                buf_x = int(r["w"] * 0.30)
                buf_y = int(r["h"] * 0.30)
                sh, sw = screen.shape[:2]
                crop_x1 = max(0, r["x"] - buf_x)
                crop_y1 = max(0, r["y"] - buf_y)
                crop_x2 = min(sw, r["x"] + r["w"] + buf_x)
                crop_y2 = min(sh, r["y"] + r["h"] + buf_y)
                crop = screen[crop_y1:crop_y2, crop_x1:crop_x2]

                if crop.size > 0:
                    florence_label = candidate.get("florence_caption", "")
                    if florence_label:
                        context_prompt = (
                            f'A fast vision model identified this element as: '
                            f'"{florence_label}". Confirm or correct the '
                            f'identification. Return JSON: '
                            f'{{"element_type": ..., "label_guess": ..., '
                            f'"confidence": 0.0-1.0, "ocr_text": ...}}'
                        )
                    else:
                        context_prompt = "Identify this UI element type and label"

                    vision_result = analyze_crop_array(crop, context_prompt)
                    if vision_result:
                        vlm_label = vision_result.get("label_guess", "")
                        vlm_type = vision_result.get("element_type", "")
                        logger.info("VLM labeled element: %s (%s)", vlm_label, vlm_type)
            except ImportError:
                logger.debug("VLM module not available for labeling")
            except RuntimeError as e:
                logger.debug("VLM labeling failed: %s", e)

        is_bbox = w > 0 and h > 0
        dialog = TagDialog(
            element_type_guess=vlm_type or (candidate.get("type_guess", "unknown") if candidate else "unknown"),
            label_guess=vlm_label or (candidate.get("label_guess", "") if candidate else ""),
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

                # Attempt OmniParser bbox refinement (for all selections)
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
            _trigger_smart_detect(
                overlay, recorded_elements, refine_mode, on_element_clicked,
            )

    def on_close() -> None:
        logger.info("Recording ended. Captured %d elements.", len(recorded_elements))
        if recorded_elements:
            _save_recording(recorded_elements, args)
        app.quit()

    # Pre-load OmniParser and Florence-2 for fast detection during recording
    try:
        from core.detection import get_detector
        get_detector()  # triggers lazy model load
    except (ImportError, RuntimeError, OSError) as e:
        logger.warning("Detection models not available: %s", e)

    try:
        from core.florence import load_model as load_florence
        load_florence()
    except (ImportError, RuntimeError, OSError) as e:
        logger.warning("Florence-2 not available: %s", e)

    overlay = OverlayController(
        on_element_clicked=on_element_clicked,
        on_mode_changed=on_mode_changed,
        on_close=on_close,
    )
    overlay.show()  # starts in PASSTHROUGH — Ctrl+R when ready to record

    return app.exec()
