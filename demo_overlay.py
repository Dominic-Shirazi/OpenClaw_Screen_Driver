"""Quick demo to see Phase 2 overlay animations live.

Run:  python demo_overlay.py
Keys: F2 = toggle ready/recording, Ctrl+Q or ESC = quit
Mouse: move along edges to see shimmer retreat
"""
from __future__ import annotations

import sys

from PyQt6.QtWidgets import QApplication

from recorder.overlay.controller import OverlayController


def main() -> None:
    app = QApplication(sys.argv)

    ctrl = OverlayController(
        on_abort=lambda: (print("Aborted"), app.quit()),
        on_save=lambda: (print("Saved"), app.quit()),
        on_state_changed=lambda s: print(f"State: {s.name}"),
    )
    ctrl.show()

    print("Overlay running!")
    print("  F2        = toggle ready (green) / recording (red)")
    print("  Ctrl+Q    = quit (save if recording)")
    print("  ESC       = quit (abort)")
    print("  Move mouse along edges to see shimmer retreat")
    print()
    print("To test scan animation, drag a rectangle while in RECORDING mode (F2 first)")

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
