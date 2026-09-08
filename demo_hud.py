"""Phase 3 HUD Demo — visual preview of tag dialog + floating toolbar.

Run:  python demo_hud.py

Shows a dark scene with:
  1. Tag dialog with typewriter fill animation + card glow
  2. Floating toolbar (draggable) with mode switching
  3. Buttons to test: typewriter, edit mode, action type switching, modes

Press Esc to quit.
"""
from __future__ import annotations

import sys

from PyQt6.QtCore import QRectF, Qt, QTimer
from PyQt6.QtGui import QBrush, QColor, QPainter
from PyQt6.QtWidgets import (
    QApplication,
    QGraphicsScene,
    QGraphicsView,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QWidget,
    QLabel,
)

from recorder.overlay.animation_clock import AnimationClock
from recorder.overlay.tag_dialog_panel import TagDialogPanel
from recorder.overlay.toolbar_panel import ToolbarMode, ToolbarPanel


class HudDemoWindow(QWidget):
    """Demo window with dark scene + control buttons."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Phase 3 HUD Demo — Tag Dialog + Toolbar")
        self.setMinimumSize(1200, 800)
        self.setStyleSheet("background: #1a1a2e;")

        layout = QVBoxLayout(self)

        # --- QGraphicsView scene (dark background) ---
        self._scene = QGraphicsScene()
        self._scene.setSceneRect(0, 0, 1100, 600)
        self._scene.setBackgroundBrush(QBrush(QColor(15, 15, 25)))

        self._view = QGraphicsView(self._scene)
        self._view.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        self._view.setStyleSheet("border: 1px solid #333; background: #0f0f19;")
        self._view.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._view.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        layout.addWidget(self._view, stretch=1)

        # --- Animation clock ---
        self._clock = AnimationClock()

        # --- Create HUD panels ---
        self._tag_dialog = TagDialogPanel(self._clock)
        self._scene.addItem(self._tag_dialog)

        self._toolbar = ToolbarPanel(self._clock, screen_w=1100, screen_h=600)
        self._scene.addItem(self._toolbar)

        # --- Control buttons ---
        btn_layout = QHBoxLayout()

        label = QLabel("Controls:")
        label.setStyleSheet("color: #aaa; font-size: 13px;")
        btn_layout.addWidget(label)

        btn_style = (
            "QPushButton { background: #2a2a4a; color: #ddd; border: 1px solid #444; "
            "border-radius: 6px; padding: 8px 16px; font-size: 12px; } "
            "QPushButton:hover { background: #3a3a5a; }"
        )

        btn_typewriter = QPushButton("Show Tag Dialog (Typewriter)")
        btn_typewriter.setStyleSheet(btn_style)
        btn_typewriter.clicked.connect(self._show_typewriter)
        btn_layout.addWidget(btn_typewriter)

        btn_edit = QPushButton("Show Tag Dialog (Edit Mode)")
        btn_edit.setStyleSheet(btn_style)
        btn_edit.clicked.connect(self._show_edit_mode)
        btn_layout.addWidget(btn_edit)

        btn_dismiss = QPushButton("Dismiss Dialog")
        btn_dismiss.setStyleSheet(btn_style)
        btn_dismiss.clicked.connect(self._dismiss_dialog)
        btn_layout.addWidget(btn_dismiss)

        btn_toolbar = QPushButton("Show Toolbar")
        btn_toolbar.setStyleSheet(btn_style)
        btn_toolbar.clicked.connect(self._show_toolbar)
        btn_layout.addWidget(btn_toolbar)

        btn_mode_rec = QPushButton("Mode: Recording")
        btn_mode_rec.setStyleSheet(btn_style)
        btn_mode_rec.clicked.connect(lambda: self._set_mode(ToolbarMode.RECORDING))
        btn_layout.addWidget(btn_mode_rec)

        btn_mode_tag = QPushButton("Mode: Tag Open")
        btn_mode_tag.setStyleSheet(btn_style)
        btn_mode_tag.clicked.connect(lambda: self._set_mode(ToolbarMode.TAG_OPEN))
        btn_layout.addWidget(btn_mode_tag)

        btn_mode_dry = QPushButton("Mode: Dry Run")
        btn_mode_dry.setStyleSheet(btn_style)
        btn_mode_dry.clicked.connect(lambda: self._set_mode(ToolbarMode.DRY_RUN))
        btn_layout.addWidget(btn_mode_dry)

        layout.addLayout(btn_layout)

        # --- Start clock ---
        self._clock.start()

        # --- Show toolbar on start ---
        QTimer.singleShot(300, self._show_toolbar)

    def _show_typewriter(self) -> None:
        """Show tag dialog with VLM typewriter fill."""
        element_rect = QRectF(300, 200, 200, 40)
        vlm_data = {
            "label": "Search Input Field",
            "caption": "A text box where users type search queries to find content",
            "action_type": "type",
            "element_type": "textbox",
        }
        self._tag_dialog.show_dialog(element_rect, vlm_data=vlm_data, edit_mode=False)

    def _show_edit_mode(self) -> None:
        """Show tag dialog in edit mode (instant fill, no typewriter)."""
        element_rect = QRectF(300, 200, 200, 40)
        vlm_data = {
            "label": "Submit Button",
            "caption": "Submits the login form with username and password",
            "action_type": "click",
            "element_type": "button",
        }
        self._tag_dialog.show_dialog(element_rect, vlm_data=vlm_data, edit_mode=True)

    def _dismiss_dialog(self) -> None:
        """Dismiss the tag dialog."""
        self._tag_dialog.dismiss()

    def _show_toolbar(self) -> None:
        """Show the floating toolbar."""
        self._toolbar.show_toolbar()

    def _set_mode(self, mode: ToolbarMode) -> None:
        """Switch toolbar mode."""
        self._toolbar.set_mode(mode)

    def keyPressEvent(self, event: object) -> None:
        """Handle Esc to quit."""
        if hasattr(event, "key") and event.key() == Qt.Key.Key_Escape:
            self.close()


def main() -> None:
    """Launch the HUD demo."""
    app = QApplication(sys.argv)
    app.setStyle("Fusion")  # Consistent look across platforms
    window = HudDemoWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
