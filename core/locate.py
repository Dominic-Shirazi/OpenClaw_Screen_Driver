"""Element location cascade for OCSD replay.

Finds UI elements on screen using a multi-stage strategy ordered by
speed and cost (cheapest first):

1. OmniParser detect+match — saved snippet → detect boxes → CLIP match (~50ms)
2. CLIP embedding — compare saved embedding against screen crops
3. OCR text match — scoped to region around expected position (~200ms)
4. VLM full scan — screenshot → LiteLLM → match by label (expensive)
5. Position fallback — blind click at recorded coordinates (no confirmation)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pyautogui

from core.capture import screenshot_full
from core.config import get_config
from core.ocr import find_text_on_screen
from core.types import ElementNotFoundError, LocateResult, Point

if TYPE_CHECKING:
    from mapper.graph import OCSDGraph

logger = logging.getLogger(__name__)


def _resolve_position_hint(
    node_data: dict,
) -> tuple[int | None, int | None, int, int]:
    """Extracts expected pixel position from node's relative_position.

    Returns:
        (hint_x, hint_y, screen_w, screen_h) — hint values are None
        if the node has no stored position.
    """
    sw, sh = pyautogui.size()
    pos = node_data.get("relative_position", {})
    x_pct = pos.get("x_pct")
    y_pct = pos.get("y_pct")
    if x_pct is not None and y_pct is not None:
        return int(x_pct * sw), int(y_pct * sh), sw, sh
    return None, None, sw, sh


def locate_element(
    graph: OCSDGraph,
    node_id: str,
    skill_id: str = "",
) -> LocateResult:
    """Locates an element on screen using the cascading strategy.

    Cascade order (fastest / cheapest first):
    1. **OmniParser detect+match** — give it the saved snippet from recording
       and approximate position → detect all UI boxes then CLIP-match, ~50 ms.
    2. **CLIP embedding** — retrieve the SAVED embedding (from recording),
       compare against crops of the current screen near the expected
       position.  Confirms OmniParser hits or finds the element independently.
    3. **OCR text match** — search ONLY within a region around the expected
       position (not full screen!) to avoid false positives from ads etc.
    4. **VLM full-screen scan** — expensive LiteLLM call, most reliable.
    5. **Position fallback** — blind click at recorded coordinates.

    Args:
        graph: The OCSDGraph containing the node data.
        node_id: The ID of the node to locate.
        skill_id: Skill name (for loading snippet PNGs).

    Returns:
        LocateResult with the screen location and match info.

    Raises:
        ElementNotFoundError: If all cascade stages fail.
        KeyError: If node_id does not exist in the graph.
    """
    node_data = graph.get_node(node_id)
    hint_x, hint_y, sw, sh = _resolve_position_hint(node_data)

    # ------------------------------------------------------------------
    # Stage 1: OmniParser detect + CLIP match
    # ------------------------------------------------------------------
    try:
        from core.capture import load_snippet
        from core.detection import get_detector

        snippet = load_snippet(skill_id, node_id[:12])
        if snippet is not None:
            screen = screenshot_full()
            cfg = get_config()
            detector = get_detector()
            result = detector.detect_and_match(
                screen, snippet, hint_x or 0, hint_y or 0,
                match_threshold=cfg.get("detection", {}).get("match_threshold", 0.7),
                search_radius=cfg.get("detection", {}).get("search_radius", 400),
            )
            if result is not None:
                logger.info(
                    "Located [%s] via OmniParser at (%d, %d) conf=%.2f",
                    node_id[:8], result.point.x, result.point.y, result.confidence,
                )
                return result
            logger.debug("OmniParser found no match for [%s]", node_id[:8])
        else:
            logger.debug("No snippet on disk for [%s], skipping Stage 1", node_id[:8])
    except ImportError:
        logger.debug("Detection module not available, skipping Stage 1")
    except Exception as e:
        logger.debug("OmniParser locate error: %s", e)

    # ------------------------------------------------------------------
    # Stage 2: CLIP embedding
    # ------------------------------------------------------------------
    try:
        from core.capture import screenshot_region
        from core.embeddings import generate_embedding, get_embedding_by_id

        saved_emb = get_embedding_by_id(node_id)
        if saved_emb is not None and hint_x is not None and hint_y is not None:
            import cv2
            import numpy as np

            search_r = 300
            rx = max(0, hint_x - search_r)
            ry = max(0, hint_y - search_r)
            rw = min(search_r * 2, sw - rx)
            rh = min(search_r * 2, sh - ry)

            crop = screenshot_region(rx, ry, rw, rh)
            if crop.size > 0:
                rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                current_emb = generate_embedding(rgb_crop)

                score = float(np.dot(saved_emb, current_emb.T).item())

                if score > 0.75:
                    logger.info(
                        "Located [%s] via CLIP at (%d, %d) score=%.3f",
                        node_id[:8], hint_x, hint_y, score,
                    )
                    return LocateResult(
                        point=Point(hint_x, hint_y),
                        confidence=min(score, 0.85),
                        method="clip",
                    )
                else:
                    logger.debug(
                        "CLIP score %.3f too low for [%s]", score, node_id[:8],
                    )
    except ImportError:
        logger.debug("CLIP/FAISS not available, skipping Stage 2")
    except Exception as e:
        logger.debug("CLIP search error: %s", e)

    # ------------------------------------------------------------------
    # Stage 3: OCR text match — scoped to region around expected position
    # ------------------------------------------------------------------
    ocr_text = node_data.get("ocr_text")
    if ocr_text:
        logger.debug("Locate [%s] via OCR: %r", node_id[:8], ocr_text)
        result = find_text_on_screen(
            ocr_text,
            hint_x=hint_x,
            hint_y=hint_y,
            search_radius=400,
        )
        if result is not None:
            logger.info(
                "Located [%s] via OCR at (%d, %d) conf=%.2f",
                node_id[:8],
                result.point.x,
                result.point.y,
                result.confidence,
            )
            return result

    # ------------------------------------------------------------------
    # Stage 4: VLM full-screen analysis (expensive, high reliability)
    # ------------------------------------------------------------------
    try:
        from core.vision import first_pass_map_array

        full_img = screenshot_full()
        candidates = first_pass_map_array(full_img)

        target_label = node_data.get("label", "").lower()

        if target_label and candidates:
            for c in candidates:
                c_label = c.get("label_guess", "").lower()
                if target_label in c_label or c_label in target_label:
                    rect = c.get("rect", {})
                    cx = rect.get("x", 0) + rect.get("w", 0) // 2
                    cy = rect.get("y", 0) + rect.get("h", 0) // 2
                    conf = c.get("confidence", 0.5)
                    logger.info(
                        "Located [%s] via VLM at (%d, %d) conf=%.2f",
                        node_id[:8], cx, cy, conf,
                    )
                    return LocateResult(
                        point=Point(cx, cy),
                        confidence=conf,
                        method="vlm",
                    )
    except ImportError:
        logger.debug("VLM module not available, skipping Stage 4")
    except Exception as e:
        logger.debug("VLM scan error: %s", e)

    # ------------------------------------------------------------------
    # Stage 5: Position fallback — blind click at recorded coordinates
    # ------------------------------------------------------------------
    if hint_x is not None and hint_y is not None:
        pos = node_data.get("relative_position", {})
        w_pct = pos.get("w_pct", 0.0)
        h_pct = pos.get("h_pct", 0.0)
        if w_pct > 0 and h_pct > 0:
            logger.warning(
                "Located [%s] via bbox center fallback at (%d, %d) "
                "bbox=%dx%d — no visual confirmation",
                node_id[:8], hint_x, hint_y, int(w_pct * sw), int(h_pct * sh),
            )
        else:
            logger.warning(
                "Located [%s] via position fallback at (%d, %d) — "
                "no visual confirmation",
                node_id[:8], hint_x, hint_y,
            )
        return LocateResult(
            point=Point(hint_x, hint_y),
            confidence=0.3,
            method="direct",
        )

    raise ElementNotFoundError(
        node_id, f"All locate stages failed for node {node_id[:8]}",
    )
