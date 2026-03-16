# Stack Research

**Domain:** Cinematic PyQt6 overlay animations, Rich TUI, FastAPI MCP endpoints (Python screen automation tool)
**Researched:** 2026-03-16
**Confidence:** HIGH (all version claims verified via PyPI or official docs)

---

## Scope

This research covers only the **new work** for this milestone. The existing stack (CLIP, FAISS, OmniParser, Tesseract, LiteLLM, NetworkX, pyautogui, mss) is already built and documented in `.planning/codebase/STACK.md`. Do not reinstall or replace any of those components.

---

## Recommended Stack

### Core Technologies

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| PyQt6 | 6.10.2 | Cinematic overlay, animation engine | Already in project. Qt's animation framework (QPropertyAnimation, QSequentialAnimationGroup, QParallelAnimationGroup) is the only viable path for Bézier-driven shimmer, scan lines, and frosted-glass HUD in a Python overlay. No viable alternative exists that matches Qt's compositing capabilities in Python. |
| Rich | 14.3.3 | Terminal UI — routine browser, loading screen, status panels, spinners | Already in project. `Live` + `Layout` + `Panel` + `Spinner` covers every V1 TUI requirement without adding a heavier framework. |
| Typer | 0.24.1 | CLI interface — `ocsd record/run/list/inspect/update/fork/hub` commands | FastAPI's creator built it; integrates with Rich natively for colored help text. Replaces argparse with type-annotated function signatures. Zero boilerplate for subcommands. |
| fastapi-mcp | 0.4.0 | Expose existing FastAPI endpoints as MCP tools automatically | Zero-configuration: three lines of code adds MCP compatibility to the existing FastAPI app. Converts OpenAPI schemas to MCP tool definitions automatically. The only library purpose-built for this exact use case. |
| questionary | 2.1.1 | Interactive keyboard-navigable prompts within Rich TUI | Built on prompt_toolkit. Provides select lists, confirmations, and text input that work inside the Rich TUI context. Rich's built-in `Prompt` lacks arrow-key navigation; questionary fills that gap. |

### Supporting Libraries (Animation-Specific)

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| PyQt6-Frameless-Window | 0.8.0 | Cross-platform frameless+transparent overlay window with Win10/Win11 acrylic blur | Use for the overlay window base class. Handles the `WA_TranslucentBackground` + `FramelessWindowHint` boilerplate correctly on both Windows and Linux, including DWM-layer blur on Win10/11. Saves significant platform-specific code. |

### Qt Animation Classes (Built Into PyQt6, No Extra Install)

| Class | Purpose | Cinematic Use |
|-------|---------|---------------|
| `QPropertyAnimation` | Interpolates a Qt property between start/end values over time with easing curves | Border glow color cycling, opacity pulses, scan line position sweep, HUD panel slide-in |
| `QSequentialAnimationGroup` | Runs animations one after another | Scan perimeter trace → bbox lock → opacity settle sequence |
| `QParallelAnimationGroup` | Runs multiple animations simultaneously | Shimmer + opacity fade running in parallel during state transitions |
| `QEasingCurve` | Controls acceleration profile of any QPropertyAnimation | `OutCubic` for snap-in, `InOutSine` for breathing glow, `OutBounce` for confirmation feedback |
| `QTimer` | Drives frame-rate-independent animation tick loops | Scan line sweep (frame counter drives Y position), donut cloud probability visualizer (redraw per frame) |
| `QPainter` + `QLinearGradient` / `QRadialGradient` | Custom paintEvent rendering | Shimmer sweep (animated gradient offset), frosted glass tint, bbox corner brackets |
| `QGraphicsBlurEffect` | Gaussian blur applied to a widget | Frosted-glass background blur on HUD tag dialog. Note: apply with `AnimationHint` flag when animating blur radius. |

### Development Tools

| Tool | Purpose | Notes |
|------|---------|-------|
| ruff 0.1+ | Linting and formatting | Already in project dev deps. No change needed. |
| pytest 7.0+ | Test runner | Already in project. New overlay tests should mock Qt event loop with `pytest-qt`. |
| pytest-qt | PyQt6 widget testing | Add to dev deps. Provides `qtbot` fixture for signal/slot testing without a display. |

---

## Installation

```bash
# Activate venv first — always
# Windows: .venv\Scripts\activate

# New runtime dependencies for this milestone
pip install "typer[all]==0.24.1"
pip install "fastapi-mcp==0.4.0"
pip install "questionary==2.1.1"
pip install "PyQt6-Frameless-Window==0.8.0"

# Already installed (verify, do not reinstall)
# PyQt6 6.10.x — pip install "PyQt6>=6.9"
# rich 14.x    — pip install "rich>=13.0"
# fastapi      — already in project

# Dev dependency additions
pip install "pytest-qt"
```

Add to `pyproject.toml` optional group `[overlay]`:
```toml
[project.optional-dependencies]
overlay = [
    "PyQt6>=6.9",
    "PyQt6-Frameless-Window>=0.8.0",
]
tui = [
    "rich>=14.0",
    "questionary>=2.1.0",
    "typer[all]>=0.24.0",
]
api = [
    "fastapi>=0.100",
    "uvicorn[standard]>=0.23",
    "fastapi-mcp>=0.4.0",
]
```

---

## Alternatives Considered

| Recommended | Alternative | When to Use Alternative |
|-------------|-------------|-------------------------|
| `fastapi-mcp` | `fastmcp` | fastmcp if building MCP server from scratch without existing FastAPI app. OCSD already has FastAPI — fastapi-mcp is the right tool. fastmcp 1.0 was absorbed into the official MCP Python SDK; it's MCP-first, not FastAPI-first. |
| `questionary` | `InquirerPy` | InquirerPy for more complex multi-step wizard flows. For OCSD's simple routine selection menus, questionary is lighter and sufficient. |
| `Typer` | `click` | click if you need full control over argument parsing internals. Typer wraps click; for OCSD's straightforward subcommand structure, Typer's type-hint syntax is faster to write and maintain. |
| `QPropertyAnimation` + `QTimer` | Qt Quick / QML | QML for animation if the app were QML-native. OCSD is PyQt6 Widgets — mixing QML for one overlay is a significant complexity cost. Stay in Widgets. |
| `PyQt6-Frameless-Window` | Manual `WA_TranslucentBackground` + ctypes DWM calls | Manual approach for maximum control or if license is a concern (PyQt6-Frameless-Window is GPLv3). On Windows, the manual Win32 path requires ~100 lines of ctypes to match what PyQt6-Frameless-Window provides. |

---

## What NOT to Use

| Avoid | Why | Use Instead |
|-------|-----|-------------|
| `Textual` | Textual is a full-screen reactive TUI framework — correct for building complete terminal apps. OCSD's TUI is a launcher/status display, not an application. Textual's async event loop would conflict with the overlay's Qt event loop. | `Rich` + `questionary` for the lightweight TUI layer needed here. |
| `tkinter` for overlay | No transparency compositing on Windows without hacks. No GPU-assisted rendering. | PyQt6 — already decided, non-negotiable. |
| `QGraphicsBlurEffect` on large screen regions | GPU-expensive. Applied to the full overlay window it will cause visible lag on 4K screens. | Apply only to small HUD panel backgrounds (< 400px wide). For full-screen overlays, use semi-transparent fill in `paintEvent` rather than blur. |
| `mcp` SDK directly | The official MCP Python SDK is verbose for a FastAPI project. Requires defining tools manually and running a separate server process. | `fastapi-mcp` which bridges existing FastAPI endpoints to MCP with three lines of code. |
| `argparse` or `click` directly | argparse has no Rich integration. click requires explicit decorator chains. | `typer` which wraps click and integrates with Rich for colored output. |

---

## Stack Patterns by Variant

**For the cinematic overlay window (frameless, transparent, always-on-top):**
- Inherit from `PyQt6-Frameless-Window`'s `FramelessWindow`
- Set `Qt.WindowType.WindowStaysOnTopHint | Qt.WindowType.Tool` flags
- Drive all animations from `QPropertyAnimation` chains; use `QTimer` only for frame-counter-based effects (scan line Y position)
- ALL overlay elements call `hide()` before any `mss` screenshot; restore after

**For state-driven glow effects (green=ready, red=recording):**
- Use `QPropertyAnimation` on a custom `border_color` Q_PROPERTY
- Implement the property getter/setter to call `update()` — Qt will call `paintEvent` automatically
- Use `QEasingCurve.Type.InOutSine` for the breathing pulse; `QEasingCurve.Type.OutCubic` for state transitions

**For the scan line sweep animation:**
- Use `QTimer` at 16ms (60 FPS) to increment a `scan_y` integer
- In `paintEvent`, draw a horizontal gradient line at `scan_y` using `QPainter` + `QLinearGradient`
- Wrap `scan_y` at widget height to loop

**For frosted-glass HUD panels:**
- Apply `QGraphicsBlurEffect(blurRadius=15)` with `BlurHint.AnimationHint` to the panel widget
- Layer a semi-transparent `QColor(20, 20, 20, 180)` fill on top in `paintEvent`
- Keep panel dimensions small (< 400px) to avoid blur performance issues

**For the MCP endpoint layer:**
- Keep existing FastAPI routes unchanged
- Add `FastApiMCP(app)` + `mcp.mount()` in `main.py` after all routes are registered
- MCP tool names auto-derive from route operation IDs — set explicit `operation_id` on every route

---

## Version Compatibility

| Package A | Compatible With | Notes |
|-----------|-----------------|-------|
| PyQt6 6.10.x | Python 3.9–3.12 | Project uses Python 3.12.10 — confirmed compatible. |
| PyQt6-Frameless-Window 0.8.0 | PyQt6 > 6.3.1 | PyQt6 6.10.x satisfies this. Requires pywin32 on Windows. |
| fastapi-mcp 0.4.0 | Python >=3.10, FastAPI (any recent) | Project uses Python 3.12 — compatible. Verify against project's FastAPI version (0.100+). |
| questionary 2.1.1 | Python >=3.9 | Compatible. Built on prompt_toolkit 3.x. |
| typer 0.24.1 | Python >=3.7, click >=8.0 | Compatible with Python 3.12. |
| Rich 14.3.3 | Python 3.8+ | Compatible. questionary renders Rich markup — versions interoperate cleanly. |

---

## Sources

- PyPI: `PyQt6` — version 6.10.2 confirmed, released January 8, 2026
- PyPI: `fastapi-mcp` — version 0.4.0, released July 28, 2025; Python >=3.10 required
- PyPI: `PyQt6-Frameless-Window` — version 0.8.0, released February 7, 2026
- PyPI: `questionary` — version 2.1.1, released August 28, 2025
- PyPI: `typer` — version 0.24.1, released February 21, 2026
- PyPI: `rich` — version 14.3.3, released February 19, 2026
- Qt official docs: [The Animation Framework](https://doc.qt.io/qt-6/animation-overview.html) — QPropertyAnimation, QSequentialAnimationGroup, QParallelAnimationGroup, QEasingCurve verified
- Qt official docs: [QGraphicsBlurEffect](https://doc.qt.io/qt-6/qgraphicsblureffect.html) — BlurHint.AnimationHint flag verified
- GitHub: [tadata-org/fastapi_mcp](https://github.com/tadata-org/fastapi_mcp) — FastAPI-MCP zero-config MCP exposure confirmed
- GitHub: [zhiyiYo/PyQt-Frameless-Window](https://github.com/zhiyiYo/PyQt-Frameless-Window) — Win10/Win11 acrylic blur support confirmed
- Rich docs: [Live Display](https://rich.readthedocs.io/en/stable/live.html) — Layout + Panel + Live confirmed
- pythonguis.com: [Animating PyQt6 widgets with QPropertyAnimation](https://www.pythonguis.com/tutorials/pyqt6-animated-widgets/) — QPropertyAnimation widget animation patterns

---
*Stack research for: OCSD cinematic overlay, Rich TUI, FastAPI MCP milestone*
*Researched: 2026-03-16*
