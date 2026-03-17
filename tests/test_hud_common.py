"""Tests for hud_common constants and card_glow painting helper."""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import inspect

import pytest
from PyQt6.QtCore import QRectF
from PyQt6.QtGui import QColor, QPainter, QPixmap


class TestHudCommonColors:
    """Verify all HUD color constants match UI-SPEC values."""

    def test_frost_bg(self) -> None:
        from recorder.overlay.hud_common import FROST_BG

        assert isinstance(FROST_BG, QColor)
        assert (FROST_BG.red(), FROST_BG.green(), FROST_BG.blue(), FROST_BG.alpha()) == (
            15, 15, 25, 200,
        )

    def test_field_bg(self) -> None:
        from recorder.overlay.hud_common import FIELD_BG

        assert isinstance(FIELD_BG, QColor)
        assert (FIELD_BG.red(), FIELD_BG.green(), FIELD_BG.blue(), FIELD_BG.alpha()) == (
            20, 20, 35, 220,
        )

    def test_accent_green(self) -> None:
        from recorder.overlay.hud_common import ACCENT_GREEN

        assert isinstance(ACCENT_GREEN, QColor)
        assert (ACCENT_GREEN.red(), ACCENT_GREEN.green(), ACCENT_GREEN.blue()) == (
            50, 200, 50,
        )

    def test_text_primary(self) -> None:
        from recorder.overlay.hud_common import TEXT_PRIMARY

        assert (TEXT_PRIMARY.red(), TEXT_PRIMARY.green(), TEXT_PRIMARY.blue(), TEXT_PRIMARY.alpha()) == (
            240, 240, 245, 230,
        )

    def test_text_secondary(self) -> None:
        from recorder.overlay.hud_common import TEXT_SECONDARY

        assert (TEXT_SECONDARY.red(), TEXT_SECONDARY.green(), TEXT_SECONDARY.blue(), TEXT_SECONDARY.alpha()) == (
            200, 200, 210, 150,
        )


class TestHudCommonSpacing:
    """Verify spacing constants."""

    def test_spacing_values(self) -> None:
        from recorder.overlay.hud_common import SPACING

        assert SPACING.xs == 4
        assert SPACING.sm == 8
        assert SPACING.md == 16
        assert SPACING.lg == 24


class TestHudCommonConstants:
    """Verify shape, z-value, timing constants."""

    def test_corner_radius(self) -> None:
        from recorder.overlay.hud_common import CORNER_RADIUS

        assert CORNER_RADIUS == 15.0

    def test_z_values(self) -> None:
        from recorder.overlay.hud_common import Z_TAG_DIALOG, Z_TOOLBAR

        assert Z_TAG_DIALOG == 100
        assert Z_TOOLBAR == 110

    def test_typewriter_timing(self) -> None:
        from recorder.overlay.hud_common import TYPEWRITER_MIN_CPS, TYPEWRITER_MAX_CPS

        assert TYPEWRITER_MIN_CPS == 20.0
        assert TYPEWRITER_MAX_CPS == 33.0

    def test_field_stylesheet_contains_rgba(self) -> None:
        from recorder.overlay.hud_common import FIELD_STYLESHEET

        assert "rgba(20, 20, 35, 220)" in FIELD_STYLESHEET


class TestCardGlow:
    """Verify card glow painting helper."""

    def test_paint_card_glow_signature(self) -> None:
        from recorder.overlay.card_glow import paint_card_glow

        sig = inspect.signature(paint_card_glow)
        params = list(sig.parameters.keys())
        assert "painter" in params
        assert "rect" in params
        assert "brightness" in params
        assert "phase" in params

    def test_edge_point_exists(self) -> None:
        from recorder.overlay.card_glow import _edge_point

        assert callable(_edge_point)

    def test_paint_card_glow_runs_without_error(self) -> None:
        from PyQt6.QtWidgets import QApplication

        app = QApplication.instance() or QApplication([])
        from recorder.overlay.card_glow import paint_card_glow

        pixmap = QPixmap(200, 200)
        painter = QPainter(pixmap)
        rect = QRectF(10, 10, 180, 180)
        paint_card_glow(painter, rect, brightness=0.8, phase=0.5)
        painter.end()
