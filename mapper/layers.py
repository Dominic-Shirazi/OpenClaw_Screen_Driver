"""OS_UI / APP_PERSISTENT / PAGE_SPECIFIC layer classification logic.

Provides an enum for UI element layers and heuristics to classify elements
based on their type and screen position.
"""

from __future__ import annotations

import enum
import logging

logger = logging.getLogger(__name__)

# Position hints that suggest OS-level UI (taskbar, system tray, etc.)
_OS_UI_POSITION_HINTS = frozenset({
    "top_left", "top_center", "top_right",
    "bottom_left", "bottom_center", "bottom_right",
})

# Element types commonly found in OS-level UI
_OS_UI_ELEMENT_TYPES = frozenset({
    "taskbar", "system_tray", "start_menu", "clock",
    "notification", "desktop_icon",
})

# Element types commonly found in persistent app chrome
_APP_PERSISTENT_ELEMENT_TYPES = frozenset({
    "menu", "menu_item", "menubar", "toolbar", "tab_bar",
    "navigation", "sidebar", "ribbon", "breadcrumb",
    "app_header", "title_bar", "status_bar",
})

# Position hints for edges of the screen (where OS UI typically lives)
_EDGE_POSITION_HINTS = frozenset({
    "top_left", "top_center", "top_right",
    "bottom_left", "bottom_center", "bottom_right",
})


class Layer(enum.Enum):
    """UI element layer classification.

    Determines the persistence and scope of a UI element:
    - OS_UI: Operating system chrome (taskbar, system tray, etc.)
    - APP_PERSISTENT: Application-level navigation that persists across pages
    - PAGE_SPECIFIC: Content that changes with page/screen navigation
    """

    OS_UI = "os_ui"
    APP_PERSISTENT = "app_persistent"
    PAGE_SPECIFIC = "page_specific"


def classify_layer(element_type: str, position_hint: str) -> Layer:
    """Classifies a UI element into a layer using heuristics.

    Uses the element's type and its screen position to determine whether
    it belongs to the OS chrome, persistent app navigation, or
    page-specific content.

    Args:
        element_type: The type of the UI element (e.g., "button", "taskbar").
        position_hint: A region string like "top_left", "center", "bottom_right".

    Returns:
        The classified Layer enum value.
    """
    element_lower = element_type.lower()
    position_lower = position_hint.lower()

    # Check for explicit OS-level element types
    if element_lower in _OS_UI_ELEMENT_TYPES:
        logger.debug(
            "Classified '%s' at '%s' as OS_UI (element type match)",
            element_type, position_hint,
        )
        return Layer.OS_UI

    # Elements at screen edges with generic types may be OS UI
    if position_lower in _EDGE_POSITION_HINTS and element_lower in {
        "button", "icon", "label", "image",
    }:
        logger.debug(
            "Classified '%s' at '%s' as OS_UI (edge position heuristic)",
            element_type, position_hint,
        )
        return Layer.OS_UI

    # Check for persistent app navigation element types
    if element_lower in _APP_PERSISTENT_ELEMENT_TYPES:
        logger.debug(
            "Classified '%s' at '%s' as APP_PERSISTENT (element type match)",
            element_type, position_hint,
        )
        return Layer.APP_PERSISTENT

    # Default: page-specific content
    logger.debug(
        "Classified '%s' at '%s' as PAGE_SPECIFIC (default)",
        element_type, position_hint,
    )
    return Layer.PAGE_SPECIFIC


def is_always_visible(layer: Layer) -> bool:
    """Determines whether elements in a given layer are always visible.

    OS_UI and APP_PERSISTENT layers are considered always-visible because
    they persist across page/screen transitions. PAGE_SPECIFIC elements
    may change or disappear during navigation.

    Args:
        layer: The Layer to check.

    Returns:
        True if elements in this layer are always visible, False otherwise.
    """
    return layer in {Layer.OS_UI, Layer.APP_PERSISTENT}
