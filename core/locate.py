"""Element location cascade for OCSD replay.

Finds UI elements on screen using a multi-stage strategy ordered by
speed and cost (cheapest first):

0. Context scoping — if a context embedding is present, locate the
   context zone on screen first, then search ONLY within that zone
   using stages 1-2.  Falls through to the full-screen cascade on
   failure.
1. OmniParser detect+match — saved snippet → detect boxes → CLIP match (~50ms)
2. CLIP embedding — compare saved embedding against screen crops
3. OCR text match — scoped to region around expected position (~200ms)
4. VLM full scan — screenshot → LiteLLM → match by label (expensive)
5. Position fallback — blind click at recorded coordinates (no confirmation)
"""

from __future__ import annotations

import logging
import re
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pyautogui

from core.capture import screenshot_full
from core.config import get_config
from core.ocr import find_text_on_screen
from core.types import ElementNotFoundError, LocateResult, Point

if TYPE_CHECKING:
    from mapper.graph import OCSDGraph

logger = logging.getLogger(__name__)

# Default max ratio between the "squashedness" of two bounding boxes.
# 3.0 means a 10:1 element won't match a 3:1, but 3:1 will match 2:1.
_DEFAULT_MAX_RATIO_DIFF = 3.0

# Maximum wall-clock seconds for the entire locate cascade.  If the
# deadline is exceeded the function returns a position fallback immediately
# rather than burning time on expensive stages (CLIP sliding window, VLM).
_CASCADE_TIMEOUT_S = 15.0


def _aspect_ratio_compatible(
    saved_w: int,
    saved_h: int,
    candidate_w: int,
    candidate_h: int,
    max_ratio_diff: float = _DEFAULT_MAX_RATIO_DIFF,
) -> bool:
    """Check if two bounding boxes have compatible aspect ratios.

    Uses the ratio-of-ratios approach: each box's aspect ratio is
    expressed as ``max(w, h) / min(w, h)`` (always >= 1), then the
    difference between the two is ``max(r1, r2) / min(r1, r2)``.

    Args:
        saved_w: Width of the saved reference snippet.
        saved_h: Height of the saved reference snippet.
        candidate_w: Width of the candidate bounding box.
        candidate_h: Height of the candidate bounding box.
        max_ratio_diff: Maximum tolerated ratio-of-ratios.

    Returns:
        True if the shapes are compatible.
    """
    saved_ratio = max(saved_w, saved_h) / max(min(saved_w, saved_h), 1)
    cand_ratio = max(candidate_w, candidate_h) / max(min(candidate_w, candidate_h), 1)
    ratio_diff = max(saved_ratio, cand_ratio) / max(min(saved_ratio, cand_ratio), 1)
    return ratio_diff <= max_ratio_diff


def _label_matches(target: str, candidate: str) -> bool:
    """Checks if target label matches candidate using word-boundary matching.

    Prevents false positives like "OK" matching "Book" or "Cookie".
    Uses case-insensitive word-boundary regex so "OK" only matches
    the standalone word "OK" in the candidate string, and vice versa.

    Args:
        target: The label we are looking for (e.g. "OK").
        candidate: The label returned by VLM (e.g. "OK button").

    Returns:
        True if either label appears as a whole word in the other.
    """
    if not target or not candidate:
        return False
    t = target.strip()
    c = candidate.strip()
    if not t or not c:
        return False
    # Exact match (case-insensitive) — fast path
    if t == c:
        return True
    # Word-boundary match: target as whole word in candidate, or vice versa
    t_pattern = re.compile(r"\b" + re.escape(t) + r"\b", re.IGNORECASE)
    if t_pattern.search(c):
        return True
    c_pattern = re.compile(r"\b" + re.escape(c) + r"\b", re.IGNORECASE)
    if c_pattern.search(t):
        return True
    return False


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

    # Reference element dimensions for aspect-ratio gating.
    # Populated from node bbox (w_pct/h_pct) or snippet image.
    pos = node_data.get("relative_position", {})
    ref_w: int | None = None
    ref_h: int | None = None
    w_pct = pos.get("w_pct", 0.0)
    h_pct = pos.get("h_pct", 0.0)
    if w_pct > 0 and h_pct > 0:
        ref_w = int(w_pct * sw)
        ref_h = int(h_pct * sh)

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
            snippet_h, snippet_w = snippet.shape[:2]
            ref_w, ref_h = snippet_w, snippet_h
            result = detector.detect_and_match(
                screen, snippet, hint_x or 0, hint_y or 0,
                match_threshold=cfg.get("detection", {}).get("match_threshold", 0.7),
                search_radius=cfg.get("detection", {}).get("search_radius", 400),
            )
            if result is not None:
                # Aspect-ratio gate: reject candidates whose shape
                # is wildly different from the saved snippet.
                if result.rect is not None and not _aspect_ratio_compatible(
                    snippet_w, snippet_h,
                    result.rect.w, result.rect.h,
                ):
                    logger.debug(
                        "OmniParser match for [%s] rejected: aspect ratio "
                        "mismatch (snippet %dx%d vs candidate %dx%d)",
                        node_id[:8], snippet_w, snippet_h,
                        result.rect.w, result.rect.h,
                    )
                else:
                    logger.info(
                        "Located [%s] via OmniParser at (%d, %d) conf=%.2f",
                        node_id[:8], result.point.x, result.point.y,
                        result.confidence,
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
                # Sliding window: scan sub-regions within the crop to
                # find the actual element position, not just confirm the
                # hint.  Window sizes approximate typical UI element
                # dimensions.
                best_score = -1.0
                best_cx, best_cy = hint_x, hint_y
                crop_h, crop_w = crop.shape[:2]
                window_sizes = [(80, 30), (120, 40), (60, 60), (160, 50)]
                stride = 80  # coarse pass; halves tile count vs stride=40

                for win_w, win_h in window_sizes:
                    if win_w > crop_w or win_h > crop_h:
                        continue
                    # Penalise windows whose aspect ratio
                    # diverges from the saved snippet.
                    ar_ok = (
                        ref_w is None
                        or ref_h is None
                        or _aspect_ratio_compatible(
                            ref_w, ref_h, win_w, win_h,
                        )
                    )
                    ar_penalty = 1.0 if ar_ok else 0.5
                    for wy in range(0, crop_h - win_h + 1, stride):
                        for wx in range(0, crop_w - win_w + 1, stride):
                            tile = crop[wy:wy + win_h, wx:wx + win_w]
                            rgb_tile = cv2.cvtColor(tile, cv2.COLOR_BGR2RGB)
                            tile_emb = generate_embedding(rgb_tile)
                            score = float(
                                np.dot(saved_emb, tile_emb.T).item(),
                            ) * ar_penalty
                            if score > best_score:
                                best_score = score
                                # Convert tile center back to screen coords
                                best_cx = rx + wx + win_w // 2
                                best_cy = ry + wy + win_h // 2

                if best_score > 0.75:
                    logger.info(
                        "Located [%s] via CLIP at (%d, %d) score=%.3f",
                        node_id[:8], best_cx, best_cy, best_score,
                    )
                    return LocateResult(
                        point=Point(best_cx, best_cy),
                        confidence=min(best_score, 0.85),
                        method="clip",
                    )
                else:
                    logger.debug(
                        "CLIP best score %.3f too low for [%s]",
                        best_score, node_id[:8],
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
                if _label_matches(target_label, c_label):
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


def _find_context_zone(
    context_emb: Any,
    screen: Any,
    sw: int,
    sh: int,
) -> tuple[int, int, int, int] | None:
    """Locate the context region on the current screen via CLIP sliding window.

    Scans the full screen with large windows to find where the saved
    context embedding best matches.  Returns the bounding box
    ``(x, y, w, h)`` in screen coordinates, or *None* if no confident
    match is found.

    Args:
        context_emb: Saved CLIP embedding for the context region (1-D ndarray).
        screen: Full-screen BGR image (numpy array).
        sw: Screen width in pixels.
        sh: Screen height in pixels.

    Returns:
        ``(x, y, w, h)`` of the best-matching zone, or *None*.
    """
    import cv2
    import numpy as np

    from core.embeddings import generate_embedding

    best_score: float = -1.0
    best_box: tuple[int, int, int, int] | None = None

    # Use larger windows than element search — context zones are
    # bigger UI regions (panels, dialogs, toolbars).
    window_sizes = [(300, 300), (400, 250), (250, 400), (500, 300), (300, 500)]
    stride = 100

    for win_w, win_h in window_sizes:
        if win_w > sw or win_h > sh:
            continue
        for wy in range(0, sh - win_h + 1, stride):
            for wx in range(0, sw - win_w + 1, stride):
                tile = screen[wy : wy + win_h, wx : wx + win_w]
                rgb_tile = cv2.cvtColor(tile, cv2.COLOR_BGR2RGB)
                tile_emb = generate_embedding(rgb_tile)
                score = float(np.dot(context_emb, tile_emb.T).item())
                if score > best_score:
                    best_score = score
                    best_box = (wx, wy, win_w, win_h)

    threshold = 0.70
    if best_score >= threshold and best_box is not None:
        logger.info(
            "Context zone found at (%d, %d, %d, %d) score=%.3f",
            *best_box,
            best_score,
        )
        return best_box

    logger.debug("Context zone not found (best score=%.3f)", best_score)
    return None


def _try_context_scoped_locate(
    step: dict[str, Any],
    routine_dir: Path,
    sw: int,
    sh: int,
    *,
    ref_w: int | None = None,
    ref_h: int | None = None,
    skip_vlm: bool = False,
) -> LocateResult | None:
    """Attempt to locate the target element within a context zone.

    If the step carries a ``context_embedding_path``, this function
    loads that embedding, finds where the context region appears on
    screen, crops to that zone, then runs OmniParser + CLIP within the
    crop.  Returned coordinates are translated back to full-screen
    space.

    Args:
        step: V1 step dictionary.
        routine_dir: Base directory for resolving relative paths.
        sw: Screen width.
        sh: Screen height.
        ref_w: Reference element width (pixels) for aspect-ratio gating.
        ref_h: Reference element height (pixels) for aspect-ratio gating.
        skip_vlm: Forwarded to inner stages (currently unused, reserved).

    Returns:
        A :class:`LocateResult` if the element is found inside the
        context zone, otherwise *None* (caller should fall through to
        the normal cascade).
    """
    context_rel = step.get("context_embedding_path")
    if not context_rel:
        return None

    ctx_path = routine_dir / context_rel
    if not ctx_path.exists():
        logger.debug("Context embedding file not found at %s", ctx_path)
        return None

    try:
        import cv2
        import numpy as np

        from core.capture import screenshot_region
        from core.embeddings import generate_embedding

        context_emb = np.load(str(ctx_path))
        logger.debug("Loaded context embedding from %s", ctx_path)

        screen = screenshot_full()
        zone = _find_context_zone(context_emb, screen, sw, sh)
        if zone is None:
            logger.debug("Context zone not located, falling through to normal cascade")
            return None

        zx, zy, zw, zh = zone

        # Add padding around the zone to avoid cutting off edge elements.
        pad = 50
        zx_padded = max(0, zx - pad)
        zy_padded = max(0, zy - pad)
        zw_padded = min(zw + 2 * pad, sw - zx_padded)
        zh_padded = min(zh + 2 * pad, sh - zy_padded)

        crop = screen[zy_padded : zy_padded + zh_padded, zx_padded : zx_padded + zw_padded]
        if crop.size == 0:
            logger.debug("Context zone crop is empty, falling through")
            return None

        # --- Sub-stage A: OmniParser within context crop ---
        snippet_rel = step.get("snippet_path")
        if snippet_rel:
            try:
                from core.detection import get_detector

                snippet_path = routine_dir / snippet_rel
                snippet = cv2.imread(str(snippet_path))
                if snippet is not None:
                    snippet_h, snippet_w = snippet.shape[:2]
                    cfg = get_config()
                    detector = get_detector()
                    # Search within the crop — hint at crop centre.
                    crop_cx = zw_padded // 2
                    crop_cy = zh_padded // 2
                    result = detector.detect_and_match(
                        crop,
                        snippet,
                        crop_cx,
                        crop_cy,
                        match_threshold=cfg.get("detection", {}).get(
                            "match_threshold", 0.7,
                        ),
                        search_radius=max(zw_padded, zh_padded),
                    )
                    if result is not None:
                        if result.rect is not None and not _aspect_ratio_compatible(
                            snippet_w, snippet_h,
                            result.rect.w, result.rect.h,
                        ):
                            logger.debug(
                                "Context-scoped OmniParser match rejected: "
                                "aspect ratio mismatch",
                            )
                        else:
                            # Translate crop-local coords to screen coords.
                            screen_x = result.point.x + zx_padded
                            screen_y = result.point.y + zy_padded
                            logger.info(
                                "Located [%s] via context+OmniParser at (%d, %d) "
                                "conf=%.2f",
                                step.get("node_id", "?")[:8],
                                screen_x,
                                screen_y,
                                result.confidence,
                            )
                            return LocateResult(
                                point=Point(screen_x, screen_y),
                                confidence=result.confidence,
                                method="context+omni",
                                rect=result.rect,
                            )
            except ImportError:
                logger.debug("Detection module not available for context-scoped Stage A")
            except Exception as e:
                logger.debug("Context-scoped OmniParser error: %s", e)

        # --- Sub-stage B: CLIP embedding within context crop ---
        embedding_rel = step.get("embedding_path")
        if embedding_rel:
            try:
                emb_path = routine_dir / embedding_rel
                if emb_path.exists():
                    saved_emb = np.load(str(emb_path))
                    crop_h, crop_w = crop.shape[:2]
                    best_score: float = -1.0
                    best_cx, best_cy = crop_w // 2, crop_h // 2
                    window_sizes = [(80, 30), (120, 40), (60, 60), (160, 50)]
                    stride = 80  # coarse pass; halves tile count vs stride=40

                    for win_w, win_h in window_sizes:
                        if win_w > crop_w or win_h > crop_h:
                            continue
                        ar_ok = (
                            ref_w is None
                            or ref_h is None
                            or _aspect_ratio_compatible(ref_w, ref_h, win_w, win_h)
                        )
                        ar_penalty = 1.0 if ar_ok else 0.5
                        for wy in range(0, crop_h - win_h + 1, stride):
                            for wx in range(0, crop_w - win_w + 1, stride):
                                tile = crop[wy : wy + win_h, wx : wx + win_w]
                                rgb_tile = cv2.cvtColor(tile, cv2.COLOR_BGR2RGB)
                                tile_emb = generate_embedding(rgb_tile)
                                score = float(
                                    np.dot(saved_emb, tile_emb.T).item(),
                                ) * ar_penalty
                                if score > best_score:
                                    best_score = score
                                    best_cx = wx + win_w // 2
                                    best_cy = wy + win_h // 2

                    if best_score > 0.75:
                        screen_x = best_cx + zx_padded
                        screen_y = best_cy + zy_padded
                        logger.info(
                            "Located [%s] via context+CLIP at (%d, %d) score=%.3f",
                            step.get("node_id", "?")[:8],
                            screen_x,
                            screen_y,
                            best_score,
                        )
                        return LocateResult(
                            point=Point(screen_x, screen_y),
                            confidence=min(best_score, 0.85),
                            method="context+clip",
                        )
                    else:
                        logger.debug(
                            "Context-scoped CLIP best score %.3f too low",
                            best_score,
                        )
            except ImportError:
                logger.debug("CLIP not available for context-scoped Stage B")
            except Exception as e:
                logger.debug("Context-scoped CLIP error: %s", e)

        logger.debug(
            "Context-scoped search found zone but no element match, "
            "falling through to normal cascade",
        )
        return None

    except ImportError:
        logger.debug("Dependencies not available for context-scoped locate")
        return None
    except Exception as e:
        logger.debug("Context-scoped locate error: %s", e)
        return None


def locate_element_from_step(
    step: dict[str, Any],
    routine_dir: Path,
    *,
    skip_vlm: bool = True,
    skip_position_fallback: bool = False,
) -> LocateResult:
    """Locates an element on screen using a v1 step dict instead of a graph node.

    Mirrors the 5-stage cascade of :func:`locate_element` but extracts
    all data (snippet path, embedding path, OCR text, position hints,
    label) from the step dictionary produced by :func:`routine.format.build_v1_step`.

    Args:
        step: A v1 step dict with keys ``anchors``, ``snippet_path``,
            ``embedding_path``, ``label``, ``element_type``, ``node_id``.
        routine_dir: Base directory for resolving relative snippet and
            embedding file paths.
        skip_vlm: If True (default), skip Stage 4 (VLM). VLM calls
            take 10+ seconds; position fallback is faster during replay.
            Pass False to enable VLM for high-reliability scenarios.
        skip_position_fallback: If True, skip Stage 5 (position).
            Used for condition checking where blind-clicking is
            never acceptable.

    Returns:
        LocateResult with the screen location and match info.

    Raises:
        ElementNotFoundError: If all enabled cascade stages fail.
    """
    cascade_t0 = time.monotonic()
    deadline = cascade_t0 + _CASCADE_TIMEOUT_S
    anchors = step.get("anchors", {})
    position_pct = anchors.get("position_pct", {})
    node_id = step.get("node_id", "unknown")
    logger.info("Locate cascade START for [%s]", node_id[:8])

    # Resolve position hint from percentage-based anchors
    sw, sh = pyautogui.size()
    x_pct = position_pct.get("x_pct")
    y_pct = position_pct.get("y_pct")
    hint_x: int | None = int(x_pct * sw) if x_pct is not None else None
    hint_y: int | None = int(y_pct * sh) if y_pct is not None else None

    # Reference element dimensions for aspect-ratio gating.
    # Populated from position_pct bbox or snippet image.
    ref_w: int | None = None
    ref_h: int | None = None
    step_w_pct = position_pct.get("w_pct", 0.0)
    step_h_pct = position_pct.get("h_pct", 0.0)
    if step_w_pct > 0 and step_h_pct > 0:
        ref_w = int(step_w_pct * sw)
        ref_h = int(step_h_pct * sh)

    # --- Deadline gate (checked before every stage) ---
    def _timed_out() -> bool:
        return time.monotonic() > deadline

    def _timeout_fallback(stage_name: str) -> LocateResult | None:
        """Return position fallback if deadline exceeded, else None."""
        if not _timed_out():
            return None
        elapsed = time.monotonic() - cascade_t0
        if hint_x is not None and hint_y is not None:
            logger.warning(
                "Locate cascade timed out before %s (%.1fs), using position fallback at (%d, %d)",
                stage_name, elapsed, hint_x, hint_y,
            )
            return LocateResult(point=Point(hint_x, hint_y), confidence=0.2, method="timeout_fallback")
        logger.warning(
            "Locate cascade timed out before %s (%.1fs) with no position hint",
            stage_name, elapsed,
        )
        return None

    # ------------------------------------------------------------------
    # Stage 0: Context-aware scoping — narrow search to context zone
    # ------------------------------------------------------------------
    t0 = time.monotonic()
    context_result = _try_context_scoped_locate(
        step, routine_dir, sw, sh,
        ref_w=ref_w, ref_h=ref_h,
        skip_vlm=skip_vlm,
    )
    elapsed = time.monotonic() - t0
    if context_result is not None:
        logger.info(
            "Stage 0 (context-scoped): %.1fs — hit via %s",
            elapsed, context_result.method,
        )
        logger.info("Locate cascade END for [%s]: %.1fs total", node_id[:8], time.monotonic() - cascade_t0)
        return context_result
    logger.info("Stage 0 (context-scoped): %.1fs — miss", elapsed)

    # ------------------------------------------------------------------
    # Stage 1: OmniParser detect + CLIP match
    # ------------------------------------------------------------------
    fb = _timeout_fallback("Stage 1")
    if fb is not None:
        return fb
    t1 = time.monotonic()
    stage1_hit = False
    snippet_rel = step.get("snippet_path")
    if snippet_rel:
        try:
            import cv2

            from core.detection import get_detector

            snippet_path = routine_dir / snippet_rel
            snippet = cv2.imread(str(snippet_path))
            if snippet is not None:
                snippet_h, snippet_w = snippet.shape[:2]
                ref_w, ref_h = snippet_w, snippet_h
                screen = screenshot_full()
                cfg = get_config()
                detector = get_detector()
                result = detector.detect_and_match(
                    screen, snippet, hint_x or 0, hint_y or 0,
                    match_threshold=cfg.get("detection", {}).get("match_threshold", 0.7),
                    search_radius=cfg.get("detection", {}).get("search_radius", 400),
                )
                if result is not None:
                    # Aspect-ratio gate: reject candidates whose shape
                    # is wildly different from the saved snippet.
                    if result.rect is not None and not _aspect_ratio_compatible(
                        snippet_w, snippet_h,
                        result.rect.w, result.rect.h,
                    ):
                        logger.debug(
                            "OmniParser match for [%s] rejected: aspect ratio "
                            "mismatch (snippet %dx%d vs candidate %dx%d)",
                            node_id[:8], snippet_w, snippet_h,
                            result.rect.w, result.rect.h,
                        )
                    else:
                        logger.info(
                            "Located [%s] via OmniParser at (%d, %d) conf=%.2f",
                            node_id[:8], result.point.x, result.point.y,
                            result.confidence,
                        )
                        stage1_hit = True
                        logger.info("Stage 1 (OmniParser): %.1fs — hit", time.monotonic() - t1)
                        logger.info("Locate cascade END for [%s]: %.1fs total", node_id[:8], time.monotonic() - cascade_t0)
                        return result
                logger.debug("OmniParser found no match for [%s]", node_id[:8])
            else:
                logger.debug("Snippet file not found at %s, skipping Stage 1", snippet_path)
        except ImportError:
            logger.debug("Detection module not available, skipping Stage 1")
        except Exception as e:
            logger.debug("OmniParser locate error: %s", e)
    if not stage1_hit:
        logger.info("Stage 1 (OmniParser): %.1fs — miss", time.monotonic() - t1)

    # ------------------------------------------------------------------
    # Stage 2: CLIP embedding
    # ------------------------------------------------------------------
    fb = _timeout_fallback("Stage 2")
    if fb is not None:
        return fb
    t2 = time.monotonic()
    stage2_hit = False
    embedding_rel = step.get("embedding_path")
    if embedding_rel and hint_x is not None and hint_y is not None:
        try:
            import cv2
            import numpy as np

            from core.capture import screenshot_region
            from core.embeddings import generate_embedding

            emb_path = routine_dir / embedding_rel
            if emb_path.exists():
                saved_emb = np.load(str(emb_path))

                search_r = 300
                rx = max(0, hint_x - search_r)
                ry = max(0, hint_y - search_r)
                rw = min(search_r * 2, sw - rx)
                rh = min(search_r * 2, sh - ry)

                crop = screenshot_region(rx, ry, rw, rh)
                if crop.size > 0:
                    # Sliding window: scan sub-regions to find actual
                    # element position rather than echoing the hint.
                    best_score = -1.0
                    best_cx, best_cy = hint_x, hint_y
                    crop_h, crop_w = crop.shape[:2]
                    window_sizes = [(80, 30), (120, 40), (60, 60), (160, 50)]
                    stride = 80  # coarse pass; halves tile count vs stride=40

                    for win_w, win_h in window_sizes:
                        if win_w > crop_w or win_h > crop_h:
                            continue
                        # Penalise windows whose aspect ratio
                        # diverges from the saved snippet.
                        ar_ok = (
                            ref_w is None
                            or ref_h is None
                            or _aspect_ratio_compatible(
                                ref_w, ref_h, win_w, win_h,
                            )
                        )
                        ar_penalty = 1.0 if ar_ok else 0.5
                        for wy in range(0, crop_h - win_h + 1, stride):
                            for wx in range(0, crop_w - win_w + 1, stride):
                                tile = crop[wy:wy + win_h, wx:wx + win_w]
                                rgb_tile = cv2.cvtColor(tile, cv2.COLOR_BGR2RGB)
                                tile_emb = generate_embedding(rgb_tile)
                                score = float(
                                    np.dot(saved_emb, tile_emb.T).item(),
                                ) * ar_penalty
                                if score > best_score:
                                    best_score = score
                                    best_cx = rx + wx + win_w // 2
                                    best_cy = ry + wy + win_h // 2

                    if best_score > 0.75:
                        logger.info(
                            "Located [%s] via CLIP at (%d, %d) score=%.3f",
                            node_id[:8], best_cx, best_cy, best_score,
                        )
                        stage2_hit = True
                        logger.info("Stage 2 (CLIP): %.1fs — hit", time.monotonic() - t2)
                        logger.info("Locate cascade END for [%s]: %.1fs total", node_id[:8], time.monotonic() - cascade_t0)
                        return LocateResult(
                            point=Point(best_cx, best_cy),
                            confidence=min(best_score, 0.85),
                            method="clip",
                        )
                    else:
                        logger.debug(
                            "CLIP best score %.3f too low for [%s]",
                            best_score, node_id[:8],
                        )
        except ImportError:
            logger.debug("CLIP/FAISS not available, skipping Stage 2")
        except Exception as e:
            logger.debug("CLIP search error: %s", e)
    if not stage2_hit:
        logger.info("Stage 2 (CLIP): %.1fs — miss", time.monotonic() - t2)

    # ------------------------------------------------------------------
    # Stage 3: OCR text match — scoped to region around expected position
    # ------------------------------------------------------------------
    fb = _timeout_fallback("Stage 3")
    if fb is not None:
        return fb
    t3 = time.monotonic()
    stage3_hit = False
    ocr_text = anchors.get("ocr_text")
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
            stage3_hit = True
            logger.info("Stage 3 (OCR): %.1fs — hit", time.monotonic() - t3)
            logger.info("Locate cascade END for [%s]: %.1fs total", node_id[:8], time.monotonic() - cascade_t0)
            return result
    if not stage3_hit:
        logger.info("Stage 3 (OCR): %.1fs — miss", time.monotonic() - t3)

    # ------------------------------------------------------------------
    # Stage 4: VLM full-screen analysis (expensive, high reliability)
    # ------------------------------------------------------------------
    fb = _timeout_fallback("Stage 4")
    if fb is not None:
        return fb
    t4 = time.monotonic()
    stage4_hit = False
    if not skip_vlm:
        try:
            from core.vision import first_pass_map_array

            full_img = screenshot_full()
            candidates = first_pass_map_array(full_img)

            target_label = step.get("label", "").lower()

            if target_label and candidates:
                for c in candidates:
                    c_label = c.get("label_guess", "").lower()
                    if _label_matches(target_label, c_label):
                        rect = c.get("rect", {})
                        cx = rect.get("x", 0) + rect.get("w", 0) // 2
                        cy = rect.get("y", 0) + rect.get("h", 0) // 2
                        conf = c.get("confidence", 0.5)
                        logger.info(
                            "Located [%s] via VLM at (%d, %d) conf=%.2f",
                            node_id[:8], cx, cy, conf,
                        )
                        stage4_hit = True
                        logger.info("Stage 4 (VLM): %.1fs — hit", time.monotonic() - t4)
                        logger.info("Locate cascade END for [%s]: %.1fs total", node_id[:8], time.monotonic() - cascade_t0)
                        return LocateResult(
                            point=Point(cx, cy),
                            confidence=conf,
                            method="vlm",
                        )
        except ImportError:
            logger.debug("VLM module not available, skipping Stage 4")
        except Exception as e:
            logger.debug("VLM scan error: %s", e)
    if not stage4_hit:
        logger.info("Stage 4 (VLM): %.1fs — %s", time.monotonic() - t4, "skipped" if skip_vlm else "miss")

    # ------------------------------------------------------------------
    # Stage 5: Position fallback — blind click at recorded coordinates
    # ------------------------------------------------------------------
    t5 = time.monotonic()
    if not skip_position_fallback and hint_x is not None and hint_y is not None:
        logger.warning(
            "Located [%s] via position fallback at (%d, %d) — "
            "no visual confirmation",
            node_id[:8], hint_x, hint_y,
        )
        logger.info("Stage 5 (position fallback): %.1fs — hit", time.monotonic() - t5)
        logger.info("Locate cascade END for [%s]: %.1fs total", node_id[:8], time.monotonic() - cascade_t0)
        return LocateResult(
            point=Point(hint_x, hint_y),
            confidence=0.3,
            method="direct",
        )
    logger.info("Stage 5 (position fallback): %.1fs — %s", time.monotonic() - t5, "skipped" if skip_position_fallback else "miss")

    logger.info("Locate cascade END for [%s]: %.1fs total — FAILED", node_id[:8], time.monotonic() - cascade_t0)
    raise ElementNotFoundError(
        node_id, f"All locate stages failed for step {node_id[:8]}",
    )
