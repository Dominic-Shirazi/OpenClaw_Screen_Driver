---
phase: 1
slug: overlay-foundation
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-16
---

# Phase 1 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 7.x |
| **Config file** | pyproject.toml `[tool.pytest.ini_options]` |
| **Quick run command** | `python -m pytest tests/test_overlay/ -x -q` |
| **Full suite command** | `python -m pytest tests/test_overlay/ -v` |
| **Estimated runtime** | ~5 seconds |

---

## Sampling Rate

- **After every task commit:** Run `python -m pytest tests/test_overlay/ -x -q`
- **After every plan wave:** Run `python -m pytest tests/test_overlay/ -v`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 5 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 01-01-01 | 01 | 1 | OVLY-01 | unit | `pytest tests/test_overlay/test_window.py -k transparent` | ❌ W0 | ⬜ pending |
| 01-01-02 | 01 | 1 | OVLY-01 | unit | `pytest tests/test_overlay/test_window.py -k click_through` | ❌ W0 | ⬜ pending |
| 01-02-01 | 02 | 1 | OVLY-02 | unit | `pytest tests/test_overlay/test_dpi.py` | ❌ W0 | ⬜ pending |
| 01-03-01 | 03 | 2 | OVLY-03 | unit | `pytest tests/test_overlay/test_capture.py -k hide_before_capture` | ❌ W0 | ⬜ pending |
| 01-03-02 | 03 | 2 | OVLY-03 | unit | `pytest tests/test_overlay/test_capture.py -k dwm_flush` | ❌ W0 | ⬜ pending |
| 01-04-01 | 04 | 2 | OVLY-04 | unit | `pytest tests/test_overlay/test_state.py` | ❌ W0 | ⬜ pending |
| 01-05-01 | 05 | 3 | OVLY-05 | integration | `pytest tests/test_overlay/test_platform.py` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/test_overlay/conftest.py` — shared fixtures (mock QApplication, screen geometry)
- [ ] `tests/test_overlay/test_window.py` — stubs for OVLY-01 (transparency, click-through)
- [ ] `tests/test_overlay/test_dpi.py` — stubs for OVLY-02 (DPI coordinate conversion)
- [ ] `tests/test_overlay/test_capture.py` — stubs for OVLY-03 (hide-before-capture, DwmFlush)
- [ ] `tests/test_overlay/test_state.py` — stubs for OVLY-04 (state machine transitions)
- [ ] `tests/test_overlay/test_platform.py` — stubs for OVLY-05 (platform detection, XCB fallback)

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Green/red shimmer visual transition | OVLY-04 | Visual quality is subjective | Toggle F2, observe smooth color transition |
| Overlay artifacts absent from screenshot | OVLY-03 | Requires real compositor + mss capture | Run capture with overlay visible, inspect output image |
| Click-through on live desktop | OVLY-01 | Requires real window manager interaction | Click through overlay onto desktop app, verify focus passes |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 5s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
