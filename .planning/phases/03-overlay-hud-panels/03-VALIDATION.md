---
phase: 3
slug: overlay-hud-panels
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-16
---

# Phase 3 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 8.x |
| **Config file** | pyproject.toml |
| **Quick run command** | `python -m pytest tests/test_tag_dialog_panel.py tests/test_toolbar_panel.py tests/test_typewriter_engine.py -x -q` |
| **Full suite command** | `python -m pytest tests/ -x -q` |
| **Estimated runtime** | ~12 seconds |

---

## Sampling Rate

- **After every task commit:** Run `python -m pytest tests/test_tag_dialog_panel.py tests/test_toolbar_panel.py tests/test_typewriter_engine.py -x -q`
- **After every plan wave:** Run `python -m pytest tests/ -x -q`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 15 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 03-01-01 | 01 | 1 | HUD-01 | unit | `python -m pytest tests/test_tag_dialog_panel.py -x` | Wave 0 | ⬜ pending |
| 03-01-02 | 01 | 1 | HUD-04 | unit | `python -m pytest tests/test_tag_dialog_panel.py::TestConditionalFields -x` | Wave 0 | ⬜ pending |
| 03-02-01 | 02 | 1 | HUD-02 | unit | `python -m pytest tests/test_typewriter_engine.py -x` | Wave 0 | ⬜ pending |
| 03-02-02 | 02 | 1 | HUD-03 | unit | `python -m pytest tests/test_tag_dialog_panel.py::TestEditMode -x` | Wave 0 | ⬜ pending |
| 03-03-01 | 03 | 2 | HUD-05 | unit | `python -m pytest tests/test_toolbar_panel.py -x` | Wave 0 | ⬜ pending |
| 03-03-02 | 03 | 2 | HUD-06 | unit | `python -m pytest tests/test_toolbar_panel.py::TestHideForCapture -x` | Wave 0 | ⬜ pending |
| 03-04-01 | 04 | 2 | HUD-ALL | integration | `python -m pytest tests/test_hud_integration.py -x` | Wave 0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/test_tag_dialog_panel.py` — stubs for HUD-01, HUD-03, HUD-04
- [ ] `tests/test_typewriter_engine.py` — stubs for HUD-02
- [ ] `tests/test_toolbar_panel.py` — stubs for HUD-05, HUD-06
- [ ] `tests/test_hud_integration.py` — stubs for cross-panel integration

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Frosted glass visual quality | HUD-01 | Subjective visual assessment | Open tag dialog on various backgrounds, confirm frost effect is readable |
| Typewriter timing feel | HUD-02 | Subjective speed assessment | Trigger VLM fill, confirm typing speed feels like fast human typist |
| Card glow synchronized with typing | HUD-02 | Visual rhythm assessment | Watch card border glow during typewriter fill, confirm flashes match keystrokes |
| Toolbar drag smoothness | HUD-05 | Tactile interaction quality | Drag toolbar around screen, confirm smooth tracking without jitter |
| QComboBox popup z-order | HUD-04 | Platform-specific rendering | Open action dropdown on Windows 11, confirm popup appears correctly |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 15s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
