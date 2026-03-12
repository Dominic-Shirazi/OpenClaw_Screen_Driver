"""VLM inference via LiteLLM proxy (OpenAI-compatible API).

Primary element type detector for OCSD. Handles all VLM-based analysis:
- analyze_crop: identify element type, label, and visible text
- confirm_action: verify an action produced the expected result
- first_pass_map: scan a full screenshot for all candidate UI elements

Routes through LiteLLM model groups defined in .env:
  OCSD_VLM_MODEL → vision group (Qwen3 VL → Gemini → local)
"""

from __future__ import annotations

import base64
import json
import logging
import re
import tempfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from core.config import get_config
from core.types import CandidateElement, ConfirmResult, Rect, VisionResult
from recorder.element_types import ElementType

logger = logging.getLogger(__name__)

# Valid element type values for validation
_VALID_ELEMENT_TYPES = {e.value for e in ElementType}


def _get_model_name() -> str:
    """Returns the configured VLM model group name."""
    config = get_config()
    return config.get("models", {}).get("vlm", "vision")


def _get_client():
    """Returns a cached OpenAI client pointing at the LiteLLM proxy."""
    from openai import OpenAI

    config = get_config()
    litellm_cfg = config.get("litellm", {})
    base_url = litellm_cfg.get("base_url", "http://localhost:4000/v1")
    api_key = litellm_cfg.get("api_key", "no-key")

    return OpenAI(base_url=base_url, api_key=api_key)


def _encode_image_to_base64(img_path: str) -> str:
    """Encodes an image file to a data URI for the OpenAI vision API."""
    path = Path(img_path)
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def _ndarray_to_tempfile(img: np.ndarray) -> str:
    """Saves a numpy array to a temp PNG and returns the path.

    Args:
        img: BGR image as numpy array.

    Returns:
        Path to the temporary PNG file.
    """
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    cv2.imwrite(tmp.name, img)
    tmp.close()
    return tmp.name


def _call_vlm(prompt: str, image_paths: list[str]) -> str:
    """Makes a chat/completions request via the LiteLLM proxy with images.

    Uses the OpenAI-compatible vision API format. Images are sent as
    base64 data URIs in the message content array.

    Args:
        prompt: The text prompt to send.
        image_paths: List of image file paths to include.

    Returns:
        The model's response text.

    Raises:
        ConnectionError: If the LiteLLM proxy is not reachable.
        RuntimeError: If the API returns an error.
    """
    model = _get_model_name()

    # Build content array: text + images
    content: list[dict] = [{"type": "text", "text": prompt}]
    for img_path in image_paths:
        data_uri = _encode_image_to_base64(img_path)
        content.append({
            "type": "image_url",
            "image_url": {"url": data_uri},
        })

    try:
        client = _get_client()
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": content}],
            max_tokens=2048,
            temperature=0.1,
        )
        return response.choices[0].message.content or ""
    except Exception as e:
        err_str = str(e)
        # Detect connection issues vs API errors
        if any(kw in err_str.lower() for kw in ("connection", "refused", "timeout", "unreachable")):
            logger.error("Cannot reach LiteLLM proxy: %s", e)
            raise ConnectionError(
                f"Cannot reach LiteLLM proxy. Is it running? Error: {e}"
            ) from e
        logger.error("VLM API error: %s", e)
        raise RuntimeError(f"VLM API error: {e}") from e


def _extract_json(text: str) -> dict[str, Any]:
    """Extracts JSON from a VLM response, handling markdown fences and noise.

    The model often wraps JSON in ```json ... ``` blocks or adds commentary
    before/after. This function is resilient to all of that.

    Args:
        text: Raw model response text.

    Returns:
        Parsed JSON dict.

    Raises:
        ValueError: If no valid JSON can be extracted.
    """
    # Strip markdown code fences
    fenced = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)

    # Try direct parse first
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Find the first { ... } block (greedy for nested objects)
    brace_match = re.search(r"\{.*\}", text, re.DOTALL)
    if brace_match:
        try:
            return json.loads(brace_match.group(0))
        except json.JSONDecodeError:
            pass

    # Find the first [ ... ] block (for array responses)
    bracket_match = re.search(r"\[.*\]", text, re.DOTALL)
    if bracket_match:
        try:
            parsed = json.loads(bracket_match.group(0))
            if isinstance(parsed, list):
                return {"items": parsed}
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not extract JSON from VLM response: {text[:200]}")


def _extract_json_array(text: str) -> list[dict[str, Any]]:
    """Extracts a JSON array from a VLM response.

    Args:
        text: Raw model response text.

    Returns:
        List of parsed JSON dicts.

    Raises:
        ValueError: If no valid JSON array can be extracted.
    """
    # Strip markdown code fences
    fenced = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)

    text = text.strip()

    # Try direct parse as array
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return parsed
    except json.JSONDecodeError:
        pass

    # Find the first [ ... ] block
    bracket_match = re.search(r"\[.*\]", text, re.DOTALL)
    if bracket_match:
        try:
            parsed = json.loads(bracket_match.group(0))
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            pass

    # Maybe it returned a single object -- try wrapping it
    try:
        result = _extract_json(text)
        if "items" in result and isinstance(result["items"], list):
            return result["items"]
        return [result]
    except ValueError:
        pass

    raise ValueError(f"Could not extract JSON array from VLM response: {text[:200]}")


def _sanitize_element_type(raw: str) -> str:
    """Normalizes an element type string to a valid ElementType value.

    Args:
        raw: The raw element type string from the VLM.

    Returns:
        A valid ElementType value string, or "unknown" if unrecognized.
    """
    normalized = raw.strip().lower().replace(" ", "_").replace("-", "_")
    if normalized in _VALID_ELEMENT_TYPES:
        return normalized
    # Fuzzy match common VLM outputs to our enum
    fuzzy_map = {
        "text_box": "textbox",
        "text_field": "textbox",
        "input": "textbox",
        "input_field": "textbox",
        "text_input": "textbox",
        "btn": "button",
        "submit": "button",
        "link": "button_nav",
        "anchor": "button_nav",
        "navigation": "button_nav",
        "nav_button": "button_nav",
        "checkbox": "toggle",
        "radio": "toggle",
        "switch": "toggle",
        "select": "dropdown",
        "combobox": "dropdown",
        "combo_box": "dropdown",
        "menu": "dropdown",
        "scroll": "scrollbar",
        "scroll_area": "scrollbar",
        "label": "read_here",
        "text": "read_here",
        "display": "read_here",
        "output": "read_here",
        "draggable": "drag_source",
        "drop_zone": "drag_target",
        "droppable": "drag_target",
        "img": "image",
        "icon": "image",
        "picture": "image",
        "dialog": "modal",
        "popup": "modal",
        "overlay": "modal",
        "toast": "notification",
        "alert": "notification",
        "snackbar": "notification",
        "banner": "notification",
    }
    return fuzzy_map.get(normalized, "unknown")


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """Clamps a float to [lo, hi]."""
    return max(lo, min(hi, value))


def analyze_crop(img_path: str, context_prompt: str) -> dict[str, Any]:
    """Analyzes a cropped image of a UI element using Qwen2-VL.

    Args:
        img_path: Path to the cropped element image.
        context_prompt: Additional context about what we're looking for
                       (e.g., "Is this a login button?").

    Returns:
        Dict with keys: element_type (str), label_guess (str),
        confidence (float 0-1), ocr_text (str|None).
    """
    prompt = f"""You are a UI element detector for screen automation software.
Analyze this UI element image and respond with ONLY a JSON object (no other text).

Context: {context_prompt}

Respond with exactly this JSON format:
{{
  "element_type": "one of: textbox, button, button_nav, toggle, tab, dropdown, scrollbar, read_here, drag_source, drag_target, image, modal, notification, unknown",
  "label_guess": "short human-readable label for this element",
  "confidence": 0.85,
  "ocr_text": "any visible text in the element, or null if none"
}}

Rules:
- element_type MUST be one of the listed values
- label_guess should be concise (2-5 words max)
- confidence is 0.0 to 1.0, how sure you are about the element_type
- ocr_text: extract ALL visible text, null if no readable text"""

    try:
        raw_response = _call_vlm(prompt, [img_path])
        parsed = _extract_json(raw_response)

        return {
            "element_type": _sanitize_element_type(
                parsed.get("element_type", "unknown")
            ),
            "label_guess": str(parsed.get("label_guess", "unknown element")),
            "confidence": _clamp(float(parsed.get("confidence", 0.5))),
            "ocr_text": parsed.get("ocr_text"),
        }
    except (ConnectionError, RuntimeError) as e:
        logger.error("VLM analyze_crop failed: %s", e)
        return {
            "element_type": "unknown",
            "label_guess": "error: VLM unavailable",
            "confidence": 0.0,
            "ocr_text": None,
        }
    except (ValueError, KeyError, TypeError) as e:
        logger.warning("Failed to parse VLM response for analyze_crop: %s", e)
        return {
            "element_type": "unknown",
            "label_guess": "error: parse failure",
            "confidence": 0.0,
            "ocr_text": None,
        }


def confirm_action(
    before_img: str, after_img: str, intended_action: str
) -> dict[str, Any]:
    """Asks the VLM whether an action produced the expected result.

    Compares before and after screenshots to determine if the intended
    action was successfully completed.

    Args:
        before_img: Path to screenshot taken before the action.
        after_img: Path to screenshot taken after the action.
        intended_action: Description of what the action was supposed to do
                        (e.g., "click the Login button to navigate to dashboard").

    Returns:
        Dict with keys: success (bool), confidence (float 0-1), notes (str).
    """
    prompt = f"""You are a UI action validator for screen automation software.
Compare these two screenshots: the first is BEFORE an action, the second is AFTER.

Intended action: {intended_action}

Determine if the action was successfully completed.

Respond with ONLY a JSON object (no other text):
{{
  "success": true,
  "confidence": 0.9,
  "notes": "brief explanation of what changed and whether it matches the intended action"
}}

Rules:
- success: true if the screen changed in a way consistent with the intended action
- confidence: 0.0 to 1.0
- notes: 1-2 sentences max
- If the screenshots look identical, success is likely false (nothing happened)
- If an error dialog or unexpected popup appeared, success is false"""

    try:
        raw_response = _call_vlm(prompt, [before_img, after_img])
        parsed = _extract_json(raw_response)

        return {
            "success": bool(parsed.get("success", False)),
            "confidence": _clamp(float(parsed.get("confidence", 0.5))),
            "notes": str(parsed.get("notes", "No notes")),
        }
    except (ConnectionError, RuntimeError) as e:
        logger.error("VLM confirm_action failed: %s", e)
        return {
            "success": False,
            "confidence": 0.0,
            "notes": f"VLM unavailable: {e}",
        }
    except (ValueError, KeyError, TypeError) as e:
        logger.warning("Failed to parse VLM response for confirm_action: %s", e)
        return {
            "success": False,
            "confidence": 0.0,
            "notes": f"Parse failure: {e}",
        }


def first_pass_map(screenshot_path: str) -> list[dict[str, Any]]:
    """Scans a full screenshot and identifies all candidate UI elements.

    This is the initial element detection pass used during recording.
    It produces bounding boxes and type guesses for everything interactive
    visible on screen.

    Args:
        screenshot_path: Path to a full-screen screenshot.

    Returns:
        List of dicts, each with: rect ({x,y,w,h}), type_guess (str),
        label_guess (str), confidence (float 0-1).
    """
    prompt = """You are a UI element detector for screen automation software.
Scan this screenshot and identify ALL interactive UI elements.

Respond with ONLY a JSON array (no other text). Each element should be:
[
  {
    "rect": {"x": 100, "y": 200, "w": 150, "h": 40},
    "type_guess": "button",
    "label_guess": "Login Button",
    "confidence": 0.9
  }
]

Element types to use: textbox, button, button_nav, toggle, tab, dropdown,
scrollbar, read_here, drag_source, drag_target, image, modal, notification, unknown

Rules:
- Include ALL interactive elements: buttons, links, inputs, toggles, tabs, dropdowns
- Include READ_HERE elements: status text, labels, display values
- rect coordinates are in pixels from top-left of the screenshot
- x, y = top-left corner of element; w, h = width and height
- Be precise with bounding boxes -- tight fit around each element
- label_guess should be a concise human-readable description (2-5 words)
- confidence is 0.0 to 1.0 for how sure you are about type_guess
- Do NOT include decorative elements, backgrounds, or borders
- Do NOT include the same element twice"""

    try:
        raw_response = _call_vlm(prompt, [screenshot_path])
        parsed_items = _extract_json_array(raw_response)

        results: list[dict[str, Any]] = []
        for item in parsed_items:
            try:
                rect_raw = item.get("rect", {})
                x = int(rect_raw.get("x", 0))
                y = int(rect_raw.get("y", 0))
                w = int(rect_raw.get("w", 0))
                h = int(rect_raw.get("h", 0))

                # Skip degenerate rects
                if w <= 0 or h <= 0:
                    continue

                results.append({
                    "rect": {"x": x, "y": y, "w": w, "h": h},
                    "type_guess": _sanitize_element_type(
                        item.get("type_guess", "unknown")
                    ),
                    "label_guess": str(item.get("label_guess", "unknown")),
                    "confidence": _clamp(float(item.get("confidence", 0.5))),
                })
            except (TypeError, ValueError) as e:
                logger.debug("Skipping malformed candidate element: %s", e)
                continue

        logger.info("first_pass_map found %d candidate elements", len(results))
        return results

    except (ConnectionError, RuntimeError) as e:
        logger.error("VLM first_pass_map failed: %s", e)
        return []
    except (ValueError, KeyError, TypeError) as e:
        logger.warning("Failed to parse VLM response for first_pass_map: %s", e)
        return []


# ---------------------------------------------------------------------------
# Convenience functions for numpy array inputs
# ---------------------------------------------------------------------------

def analyze_crop_array(img: np.ndarray, context_prompt: str) -> dict[str, Any]:
    """Like analyze_crop, but accepts a numpy array instead of a file path.

    Args:
        img: BGR image as numpy array.
        context_prompt: Additional context about the element.

    Returns:
        Same as analyze_crop.
    """
    tmp_path = _ndarray_to_tempfile(img)
    try:
        return analyze_crop(tmp_path, context_prompt)
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def first_pass_map_array(screenshot: np.ndarray) -> list[dict[str, Any]]:
    """Like first_pass_map, but accepts a numpy array instead of a file path.

    Args:
        screenshot: BGR full-screen screenshot as numpy array.

    Returns:
        Same as first_pass_map.
    """
    tmp_path = _ndarray_to_tempfile(screenshot)
    try:
        return first_pass_map(tmp_path)
    finally:
        Path(tmp_path).unlink(missing_ok=True)
