"""Shared HUD constants: colors, spacing, fonts, field stylesheet.

Provides the visual language shared across all HUD elements (tag dialog,
toolbar, typewriter engine).  Values are sourced from the Phase 3 UI-SPEC
design contract and must not be changed without updating the spec.
"""
from __future__ import annotations

import logging
import sys
from types import SimpleNamespace

from PyQt6.QtGui import QColor

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Colors (as QColor instances)
# ---------------------------------------------------------------------------

FROST_BG: QColor = QColor(15, 15, 25, 200)
"""Frosted glass panel background fill."""

FIELD_BG: QColor = QColor(20, 20, 35, 220)
"""Recessed input field wells."""

ACCENT_GREEN: QColor = QColor(50, 200, 50)
"""Accent color for glow, borders, confirm button."""

TEXT_PRIMARY: QColor = QColor(240, 240, 245, 230)
"""Input text color."""

TEXT_SECONDARY: QColor = QColor(200, 200, 210, 150)
"""Labels, placeholders, helper tips."""

TEXT_DISABLED: QColor = QColor(140, 140, 150, 80)
"""Disabled text."""

BORDER_SUBTLE: QColor = QColor(50, 200, 50, 60)
"""Input field borders (1px)."""

HIGHLIGHT_TOP: QColor = QColor(255, 255, 255, 12)
"""Frosted glass inner highlight at top edge."""

CONFIRM_BG: QColor = QColor(50, 200, 50, 180)
"""Confirm button fill."""

DISMISS_BG: QColor = QColor(60, 60, 70, 150)
"""Dismiss button fill."""

DISMISS_TEXT: QColor = QColor(180, 180, 190, 180)
"""Dismiss button text."""

# ---------------------------------------------------------------------------
# Spacing
# ---------------------------------------------------------------------------

SPACING: SimpleNamespace = SimpleNamespace(xs=4, sm=8, md=16, lg=24)
"""Spacing tokens (all multiples of 4)."""

# ---------------------------------------------------------------------------
# Shape
# ---------------------------------------------------------------------------

CORNER_RADIUS: float = 15.0
"""Panel corner radius."""

TOOLBAR_CORNER_RADIUS: float = 20.0
"""Toolbar pill radius (half height)."""

# ---------------------------------------------------------------------------
# Typography
# ---------------------------------------------------------------------------

FONT_FAMILY: str = "Segoe UI" if sys.platform == "win32" else ""
"""Font family: Segoe UI on Windows, system sans-serif fallback otherwise."""

FONT_SIZE_HELPER: int = 10
"""Helper tips font size."""

FONT_SIZE_LABEL: int = 12
"""Field labels, toolbar buttons font size."""

FONT_SIZE_INPUT: int = 14
"""Input text, section headings font size."""

FONT_WEIGHT_LIGHT: int = 300
"""Light/thin font weight."""

FONT_WEIGHT_REGULAR: int = 400
"""Regular font weight."""

# ---------------------------------------------------------------------------
# Shared field stylesheet
# ---------------------------------------------------------------------------

FIELD_STYLESHEET: str = (
    "background: rgba(20, 20, 35, 220); "
    "color: rgba(240, 240, 245, 230); "
    "border: 1px solid rgba(50, 200, 50, 60); "
    "border-radius: 4px; "
    "padding: 4px 6px; "
    "font-weight: 300; "
    "font-size: 13px;"
)
"""CSS string for QLineEdit/QComboBox styling."""

# ---------------------------------------------------------------------------
# Z-values
# ---------------------------------------------------------------------------

Z_TAG_DIALOG: int = 100
"""Tag dialog z-value (above all animation layers)."""

Z_TOOLBAR: int = 110
"""Toolbar z-value (above tag dialog)."""

# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------

FADE_IN_MS: int = 200
"""Panel fade-in duration (milliseconds)."""

FADE_OUT_MS: int = 150
"""Panel fade-out duration (milliseconds)."""

MODE_SWITCH_MS: int = 150
"""Toolbar button swap duration (milliseconds)."""

TYPEWRITER_MIN_CPS: float = 20.0
"""Minimum typewriter speed in characters per second (50ms/char)."""

TYPEWRITER_MAX_CPS: float = 33.0
"""Maximum typewriter speed in characters per second (30ms/char)."""

CURSOR_BLINK_MS: int = 500
"""Cursor blink interval (milliseconds)."""
