---
phase: 4
slug: record-flow
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-18
---

# Phase 4 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 7.x |
| **Config file** | pyproject.toml |
| **Quick run command** | `python -m pytest tests/ -x -q --tb=short` |
| **Full suite command** | `python -m pytest tests/ -v --tb=long` |
| **Estimated runtime** | ~15 seconds |

---

## Sampling Rate

- **After every task commit:** Run `python -m pytest tests/ -x -q --tb=short`
- **After every plan wave:** Run `python -m pytest tests/ -v --tb=long`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 15 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 04-01-01 | 01 | 1 | REC-01 | unit | `pytest tests/test_record_session.py -k "test_naming"` | ❌ W0 | ⬜ pending |
| 04-01-02 | 01 | 1 | REC-02 | unit | `pytest tests/test_record_session.py -k "test_f2_toggle"` | ❌ W0 | ⬜ pending |
| 04-02-01 | 02 | 1 | REC-03 | unit | `pytest tests/test_capture_pipeline.py -k "test_hide_overlay"` | ❌ W0 | ⬜ pending |
| 04-02-02 | 02 | 1 | REC-04 | unit | `pytest tests/test_capture_pipeline.py -k "test_bbox_refinement"` | ❌ W0 | ⬜ pending |
| 04-03-01 | 03 | 2 | REC-05, REC-06 | unit | `pytest tests/test_tag_dialog.py -k "test_vlm_autofill"` | ❌ W0 | ⬜ pending |
| 04-03-02 | 03 | 2 | REC-07 | unit | `pytest tests/test_tag_dialog.py -k "test_manual_fallback"` | ❌ W0 | ⬜ pending |
| 04-04-01 | 04 | 2 | REC-08, REC-09 | unit | `pytest tests/test_dryrun.py -k "test_countdown"` | ❌ W0 | ⬜ pending |
| 04-04-02 | 04 | 2 | REC-10 | unit | `pytest tests/test_dryrun.py -k "test_validation"` | ❌ W0 | ⬜ pending |
| 04-05-01 | 05 | 3 | REC-11 | unit | `pytest tests/test_routine_save.py -k "test_save_ctrlq"` | ❌ W0 | ⬜ pending |
| 04-05-02 | 05 | 3 | REC-11 | unit | `pytest tests/test_routine_save.py -k "test_abort_esc"` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/test_record_session.py` — stubs for REC-01, REC-02
- [ ] `tests/test_capture_pipeline.py` — stubs for REC-03, REC-04
- [ ] `tests/test_tag_dialog.py` — stubs for REC-05, REC-06, REC-07
- [ ] `tests/test_dryrun.py` — stubs for REC-08, REC-09, REC-10
- [ ] `tests/test_routine_save.py` — stubs for REC-11

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Overlay hidden during screenshot | REC-03 | Requires live Win32 overlay | Start recording, click element, verify screenshot has no overlay artifacts |
| Countdown visual animation | REC-08 | Requires Qt rendering verification | Start dry-run, verify 3-2-1 countdown renders and mouse remains unlocked |
| Drag-highlight bbox tightening | REC-04 | Requires live screen + AI detection | Drag rough selection, verify AI tightens to element boundary |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 15s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
