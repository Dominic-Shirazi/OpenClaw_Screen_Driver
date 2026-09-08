"""Diagnostic: check coordinate space mismatch between mss and Qt overlay.

Run: python diag_overlay.py
"""
from __future__ import annotations

import sys


def main() -> None:
    # 1. Check mss capture dimensions
    import mss
    with mss.mss() as sct:
        monitor = sct.monitors[1]
        mss_w = monitor["width"]
        mss_h = monitor["height"]
        print(f"mss monitor[1]: {mss_w}x{mss_h}")

    # 2. Check pyautogui dimensions
    import pyautogui
    pag_w, pag_h = pyautogui.size()
    print(f"pyautogui.size(): {pag_w}x{pag_h}")

    # 3. Check Qt dimensions
    from PyQt6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication(sys.argv)
    screen = QApplication.primaryScreen()
    if screen:
        geom = screen.geometry()
        qt_w, qt_h = geom.width(), geom.height()
        print(f"Qt primaryScreen.geometry(): {qt_w}x{qt_h}")

        dpr = screen.devicePixelRatio()
        print(f"Qt devicePixelRatio: {dpr}")

        phys = screen.size()
        print(f"Qt screen.size(): {phys.width()}x{phys.height()}")
    else:
        print("No Qt primary screen found")

    # 4. Check Windows DPI
    if sys.platform == "win32":
        import ctypes
        try:
            dpi = ctypes.windll.user32.GetDpiForSystem()
            scale = dpi / 96.0
            print(f"Win32 system DPI: {dpi} (scale={scale:.0%})")
        except Exception as e:
            print(f"Win32 DPI check failed: {e}")

    # Summary
    if mss_w != qt_w or mss_h != qt_h:
        ratio_w = mss_w / qt_w
        ratio_h = mss_h / qt_h
        print(f"\n*** MISMATCH! mss captures {mss_w}x{mss_h} but Qt overlay is {qt_w}x{qt_h}")
        print(f"    Scale factor: {ratio_w:.2f}x{ratio_h:.2f}")
        print(f"    Detection coords would be {ratio_w:.0f}x too large for overlay scene")
        print(f"    Boxes would appear off-screen or way too big")
    else:
        print(f"\nCoordinate spaces match: {mss_w}x{mss_h}")
        print("DPI is NOT the issue — look elsewhere for the rendering bug")


if __name__ == "__main__":
    main()
