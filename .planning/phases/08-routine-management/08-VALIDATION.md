---
phase: 8
slug: routine-management
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-20
---

# Phase 8 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 7.x |
| **Config file** | pyproject.toml |
| **Quick run command** | `python -m pytest tests/test_routine_management.py -x -q` |
| **Full suite command** | `python -m pytest tests/test_routine_management.py tests/test_update_session.py -x -q` |
| **Estimated runtime** | ~10 seconds |

---

## Sampling Rate

- **After every task commit:** Run `python -m pytest tests/test_routine_management.py -x -q`
- **After every plan wave:** Run `python -m pytest tests/test_routine_management.py tests/test_update_session.py -x -q`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 10 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 08-01-01 | 01 | 1 | MGMT-03, MGMT-04 | unit | `pytest tests/test_routine_management.py -k "delete or inspect" -x -q` | W0 | pending |
| 08-01-02 | 01 | 1 | MGMT-02 | unit | `pytest tests/test_routine_management.py -k "fork" -x -q` | W0 | pending |
| 08-02-01 | 02 | 2 | MGMT-01 | unit | `pytest tests/test_update_session.py -x -q` | W0 | pending |
| 08-02-02 | 02 | 2 | MGMT-01 | unit | `pytest tests/test_update_session.py -k "toolbar or save" -x -q` | W0 | pending |

*Status: pending / green / red / flaky*

---

## Wave 0 Requirements

- [ ] `tests/test_routine_management.py` — stubs for delete, inspect, fork
- [ ] `tests/test_update_session.py` — stubs for update walkthrough session

*Existing test infrastructure (pytest, conftest) covers framework needs.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Update walkthrough with overlay | MGMT-01 | Requires Qt overlay and screen interaction | Load a routine, walk through steps, verify toolbar controls work |
| Fork from update mid-point | MGMT-02 | Requires interactive overlay session | During update, press Fork Here at step N, verify new routine has steps 1..N |
| Rich table inspect output | MGMT-04 | Visual formatting quality | Run inspect on a routine, verify table renders correctly in terminal |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 10s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
