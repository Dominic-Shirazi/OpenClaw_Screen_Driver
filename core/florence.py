"""Florence-2 captioning module.

Fast element labeling using Microsoft Florence-2 vision model.
Provides caption_crop(), caption_batch(), and describe_element()
for UI element identification.

Model weights auto-download from HuggingFace on first use.
"""

from __future__ import annotations

import logging
import re
import threading
from pathlib import Path
from typing import Any

import numpy as np

from core import model_cache
from core.config import get_config

logger = logging.getLogger(__name__)

# Module-level state (lazy-loaded)
_model: Any = None
_processor: Any = None
_device: str = "cpu"
_lock = threading.Lock()

# Lazy imports — only needed when model loads
AutoModelForCausalLM: Any = None
AutoProcessor: Any = None

# Known UI element type keywords for describe_element()
_TYPE_KEYWORDS = [
    "button", "textbox", "text field", "input field", "input box",
    "checkbox", "radio", "toggle", "switch",
    "dropdown", "select", "combobox",
    "slider", "scrollbar", "scroll bar",
    "tab", "menu", "menubar", "toolbar",
    "link", "hyperlink", "anchor",
    "icon", "image", "logo",
    "label", "heading", "title",
    "dialog", "modal", "popup",
    "list", "table", "grid",
]


def _import_transformers():
    """Lazy-import transformers classes."""
    global AutoModelForCausalLM, AutoProcessor
    if AutoModelForCausalLM is None:
        from transformers import AutoModelForCausalLM as _M, AutoProcessor as _P
        AutoModelForCausalLM = _M
        AutoProcessor = _P


def load_model(variant: str | None = None) -> None:
    """Pre-load Florence-2 model. Called at startup for warm cache.

    Args:
        variant: Model name override. Default from config
                 (florence-2-large or florence-2-base).
    """
    global _model, _processor, _device

    with _lock:
        if _model is not None:
            return

        _import_transformers()

        cfg = get_config()
        if variant is None:
            variant = cfg.get("models", {}).get(
                "florence", "microsoft/Florence-2-large"
            )

        model_dir = model_cache.ensure_model(variant)
        logger.info("Loading Florence-2 from %s", model_dir)

        # Determine device
        _device = "cpu"
        try:
            import torch
            gpu_id = cfg.get("hardware", {}).get("gpu_embeddings", 1)
            if torch.cuda.is_available() is True:
                _device = f"cuda:{gpu_id}"
        except ImportError:
            pass

        _model = AutoModelForCausalLM.from_pretrained(
            str(model_dir), trust_remote_code=True,
        )
        _processor = AutoProcessor.from_pretrained(
            str(model_dir), trust_remote_code=True,
        )

        if _device != "cpu":
            _model = _model.to(_device)

        logger.info("Florence-2 loaded on %s", _device)


def _ensure_model() -> None:
    """Ensure model is loaded before use."""
    if _model is None:
        load_model()


def caption_crop(image: np.ndarray, task: str = "<CAPTION>") -> str:
    """Caption a single image crop.

    Args:
        image: RGB numpy array of element crop.
        task: Florence-2 task token. Options:
              "<CAPTION>" — short caption
              "<DETAILED_CAPTION>" — detailed description
              "<MORE_DETAILED_CAPTION>" — very detailed

    Returns:
        Caption string.
    """
    _ensure_model()

    from PIL import Image
    pil_img = Image.fromarray(image)

    inputs = _processor(text=task, images=pil_img, return_tensors="pt")
    # Move to device
    for k in inputs:
        if hasattr(inputs[k], "to"):
            inputs[k] = inputs[k].to(_device)

    generated_ids = _model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=128,
        num_beams=3,
    )

    generated_text = _processor.batch_decode(
        generated_ids, skip_special_tokens=False,
    )[0]

    parsed = _processor.post_process_generation(
        generated_text, task=task,
        image_size=(pil_img.width, pil_img.height),
    )

    caption = parsed.get(task, generated_text)
    if isinstance(caption, dict):
        caption = str(caption)

    return caption.strip()


def caption_batch(
    images: list[np.ndarray], task: str = "<CAPTION>"
) -> list[str]:
    """Caption multiple crops. Returns list of caption strings.

    If a crop is degenerate (empty, 0-dim), returns empty string for that
    position to maintain 1:1 correspondence with input list.

    Args:
        images: List of RGB numpy arrays.
        task: Florence-2 task token.

    Returns:
        List of caption strings, one per input image.
    """
    _ensure_model()

    results: list[str] = []
    for img in images:
        if img.size == 0 or img.ndim < 2:
            results.append("")
            continue
        try:
            results.append(caption_crop(img, task))
        except Exception as e:
            logger.warning("Florence-2 caption failed for crop: %s", e)
            results.append("")

    return results


def describe_element(image: np.ndarray) -> dict[str, str]:
    """Structured element description for UI labeling.

    Uses <DETAILED_CAPTION> internally, then parses the response
    to extract type and label.

    Args:
        image: RGB numpy array of element crop.

    Returns:
        Dict with keys: type_guess, label_guess, description.
    """
    caption = caption_crop(image, "<DETAILED_CAPTION>")

    type_guess = "unknown"
    label_guess = caption

    caption_lower = caption.lower()
    for keyword in _TYPE_KEYWORDS:
        if keyword in caption_lower:
            # Map multi-word keywords to canonical types
            if keyword in ("text field", "input field", "input box"):
                type_guess = "textbox"
            elif keyword in ("dropdown", "select", "combobox"):
                type_guess = "dropdown"
            elif keyword in ("link", "hyperlink", "anchor"):
                type_guess = "link"
            elif keyword in ("scroll bar",):
                type_guess = "scrollbar"
            elif keyword in ("switch",):
                type_guess = "toggle"
            else:
                type_guess = keyword

            # Extract label: text before or around the keyword
            # Try to find quoted text first
            quoted = re.findall(r'["\']([^"\']+)["\']', caption)
            if quoted:
                label_guess = quoted[0]
            else:
                # Use the caption but shortened
                label_guess = caption[:50]
            break

    return {
        "type_guess": type_guess,
        "label_guess": label_guess,
        "description": caption,
    }
