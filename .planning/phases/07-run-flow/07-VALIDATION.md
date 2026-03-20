---
phase: 7
slug: run-flow
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-19
---

# Phase 7 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 7.x |
| **Config file** | pyproject.toml |
| **Quick run command** | `python -m pytest tests/test_routine_runner.py -x -q` |
| **Full suite command** | `python -m pytest tests/test_routine_runner.py tests/test_condition_engine.py tests/test_locate_adapter.py tests/test_replay_overlay.py -x -q` |
| **Estimated runtime** | ~15 seconds |

---

## Sampling Rate

- **After every task commit:** Run `python -m pytest tests/test_routine_runner.py -x -q`
- **After every plan wave:** Run `python -m pytest tests/test_routine_runner.py tests/test_condition_engine.py tests/test_locate_adapter.py tests/test_replay_overlay.py -x -q`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 15 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 07-01-01 | 01 | 1 | RUN-02 | unit | `pytest tests/test_locate_adapter.py tests/test_condition_engine.py -x -q` | W0 | pending |
| 07-01-02 | 01 | 1 | RUN-02 | unit | `pytest tests/test_locate_adapter.py tests/test_condition_engine.py -x -q` | W0 | pending |
| 07-02-01 | 02 | 1 | RUN-09 | unit | `pytest tests/test_replay_overlay.py -k "replaying or shimmer or config" -x -q` | W0 | pending |
| 07-02-02 | 02 | 1 | RUN-09 | unit | `pytest tests/test_replay_overlay.py -k "badge or highlight or flash" -x -q` | W0 | pending |
| 07-03-01 | 03 | 2 | RUN-01, RUN-07 | unit | `pytest tests/test_routine_runner.py -k "run_log or prune" -x -q` | W0 | pending |
| 07-03-02 | 03 | 2 | RUN-03, RUN-04, RUN-05, RUN-06, RUN-07, RUN-08 | unit | `pytest tests/test_routine_runner.py -k "run_routine or dispatch or cascade or blind" -x -q` | W0 | pending |
| 07-03-03 | 03 | 2 | RUN-03, RUN-08 | unit | `pytest tests/test_routine_runner.py -k "callback or screenshot_taken" -x -q` | W0 | pending |
| 07-04-01 | 04 | 3 | RUN-09 | unit | `pytest tests/test_replay_overlay.py -x -q` | W0 | pending |
| 07-04-02 | 04 | 3 | RUN-09 | unit | `python -c "from routine.replay_overlay import ReplayOverlayAdapter; print('OK')"` | W0 | pending |

*Status: pending / green / red / flaky*

---

## Wave 0 Requirements

- [ ] `tests/test_routine_runner.py` — stubs for RUN-01 through RUN-08
- [ ] `tests/test_condition_engine.py` — extend existing with element_appears and text_matches tests
- [ ] `tests/test_locate_adapter.py` — stubs for locate_element_from_step tests
- [ ] `tests/test_replay_overlay.py` — stubs for overlay state, widgets, and camera flash
- [ ] `rapidfuzz` — add to pyproject.toml dependencies

*Existing test infrastructure (pytest, conftest) covers framework needs.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Purple shimmer overlay during replay | RUN-09 | Requires visual Qt overlay on display | Launch replay with `replay.show_overlay: true`, verify purple border |
| Status badge shows step info | RUN-09 | Visual UI element | Verify top-center pill shows "Step N/M: Label" during replay |
| Target bbox highlight before click | RUN-09 | Visual animation timing | Verify ~300ms purple glow around target before mouse moves |
| Camera flash after screenshot | RUN-09 | Visual overlay timing | Verify 15px border flash after each screenshot capture |
| Mouse Bezier movement is visible | RUN-03 | Visual path inspection | Watch mouse path during replay, verify curved non-linear movement |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 15s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
