---
phase: 4
slug: record-flow
status: draft
nyquist_compliant: true
wave_0_complete: false
created: 2026-03-18
---

# Phase 4 -- Validation Strategy

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

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | Test File | Status |
|---------|------|------|-------------|-----------|-------------------|-----------|--------|
| 04-01-T1 | 01 | 1 | REC-03, REC-08, REC-09 | unit | `pytest tests/test_record_phase.py tests/test_countdown_widget.py -x -q` | tests/test_record_phase.py, tests/test_countdown_widget.py | pending |
| 04-01-T2 | 01 | 1 | REC-03, REC-08, REC-09 | unit | `pytest tests/test_record_phase.py tests/test_countdown_widget.py -x -q` | tests/test_record_phase.py, tests/test_countdown_widget.py | pending |
| 04-02-T1 | 02 | 2 | REC-04, REC-05, REC-06, REC-09 | unit | `pytest tests/test_overlay_extensions.py -x -q` | tests/test_overlay_extensions.py | pending |
| 04-02-T2 | 02 | 2 | REC-04, REC-05, REC-06, REC-09 | unit | `pytest tests/test_overlay_extensions.py -x -q` | tests/test_overlay_extensions.py | pending |
| 04-03-T1 | 03 | 2 | REC-01 thru REC-07, REC-10 | unit | `pytest tests/test_record_session.py -x -q` | tests/test_record_session.py | pending |
| 04-03-T2 | 03 | 2 | REC-01 thru REC-07, REC-10 | unit | `pytest tests/test_record_session.py -x -q` | tests/test_record_session.py | pending |
| 04-04-T1 | 04 | 3 | REC-08, REC-09, REC-10, REC-11 | unit | `pytest tests/test_record_session.py tests/test_record_flow.py -x -q` | tests/test_record_session.py, tests/test_record_flow.py | pending |
| 04-04-T2 | 04 | 3 | REC-01, REC-02, REC-11 | unit | `pytest tests/test_record_flow.py -x -q` | tests/test_record_flow.py | pending |

*Status: pending / green / red / flaky*

---

## Wave 0 Requirements

Wave 0 test stubs are created inline with each plan's Task 2. No separate Wave 0 plan needed.

- [ ] `tests/test_record_phase.py` -- Plan 01 Task 2 creates (RecordPhase, PipelineBridge)
- [ ] `tests/test_countdown_widget.py` -- Plan 01 Task 2 creates (CountdownWidget, AbortPanel)
- [ ] `tests/test_overlay_extensions.py` -- Plan 02 Task 2 creates (controller extensions, bbox editing)
- [ ] `tests/test_record_session.py` -- Plan 03 Task 2 creates (RecordSession pipeline, card glow, drag reject, dry-run stages)
- [ ] `tests/test_record_flow.py` -- Plan 04 Task 2 creates (cmd_record, TUI prompt, session completion callback)

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Overlay hidden during screenshot | REC-03 | Requires live Win32 overlay | Start recording, click element, verify screenshot has no overlay artifacts |
| Countdown visual animation | REC-08 | Requires Qt rendering verification | Start dry-run, verify 3-2-1 countdown renders and mouse remains unlocked |
| Drag-highlight bbox tightening | REC-04 | Requires live screen + AI detection | Drag rough selection, verify AI tightens to element boundary |
| Card glow pulsing during detection | REC-06 | Requires visual confirmation | Click element, verify card glow pulses during DETECTING and VLM_ANALYZING phases |

---

## Validation Sign-Off

- [x] All tasks have `<automated>` verify or Wave 0 dependencies
- [x] Sampling continuity: no 3 consecutive tasks without automated verify
- [x] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [x] Feedback latency < 15s
- [x] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
