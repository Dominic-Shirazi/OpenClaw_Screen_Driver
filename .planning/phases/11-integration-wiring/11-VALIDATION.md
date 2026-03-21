---
phase: 11
slug: integration-wiring
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-21
---

# Phase 11 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 7.x |
| **Config file** | pyproject.toml |
| **Quick run command** | `python -m pytest tests/ -x -q --timeout=10` |
| **Full suite command** | `python -m pytest tests/ -q --timeout=30` |
| **Estimated runtime** | ~15 seconds |

---

## Sampling Rate

- **After every task commit:** Run `python -m pytest tests/ -x -q --timeout=10`
- **After every plan wave:** Run `python -m pytest tests/ -q --timeout=30`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 15 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 11-01-01 | 01 | 1 | SEC-01 | unit | `python -m pytest tests/test_scanner_routine.py -x -q` | ❌ W0 | ⬜ pending |
| 11-01-02 | 01 | 1 | SEC-01 | unit | `python -m pytest tests/test_scanner_routine.py -x -q` | ❌ W0 | ⬜ pending |
| 11-02-01 | 02 | 1 | RUN-09 | integration | `python -m pytest tests/test_replay_wiring.py -x -q` | ❌ W0 | ⬜ pending |
| 11-02-02 | 02 | 1 | RUN-09 | integration | `python -m pytest tests/test_replay_wiring.py -x -q` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/test_scanner_routine.py` — stubs for SEC-01 (scan_routine wrapper, block on threat)
- [ ] `tests/test_replay_wiring.py` — stubs for RUN-09 (overlay adapter instantiation in CLI/TUI)

*Existing test infrastructure (pytest, conftest) already in place.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Purple shimmer visible during CLI run | RUN-09 | Requires visual Qt overlay on screen | Run `ocsd run "test"` and observe overlay |
| StatusBadge shows current step | RUN-09 | Visual verification | Watch overlay during multi-step routine |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 15s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
