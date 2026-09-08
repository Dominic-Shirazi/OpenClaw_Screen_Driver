"""Take a screenshot and save to debug_screenshots/latest.png (overwrites each time)."""
from __future__ import annotations

import sys
import time
from pathlib import Path

def snap(delay: float = 0.0) -> Path:
    """Capture screen and save to debug_screenshots/latest.png."""
    if delay > 0:
        time.sleep(delay)
    # Use mss for fast, cross-platform screenshots
    import mss
    out = Path(__file__).parent / "debug_screenshots" / "latest.png"
    out.parent.mkdir(exist_ok=True)
    with mss.mss() as sct:
        sct.shot(mon=0, output=str(out))
    print(f"Saved: {out}")
    return out

if __name__ == "__main__":
    delay = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0
    snap(delay)
