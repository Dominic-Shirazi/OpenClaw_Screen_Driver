---
phase: 9
slug: tui-and-cli
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-20
---

# Phase 9 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 7.x |
| **Config file** | pyproject.toml |
| **Quick run command** | `python -m pytest tests/test_cli.py -x -q` |
| **Full suite command** | `python -m pytest tests/test_cli.py tests/test_tui.py -x -q` |
| **Estimated runtime** | ~10 seconds |

---

## Sampling Rate

- **After every task commit:** Run `python -m pytest tests/test_cli.py -x -q`
- **After every plan wave:** Run `python -m pytest tests/test_cli.py tests/test_tui.py -x -q`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 10 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 09-01-01 | 01 | 1 | CLI-01..09 | unit | `pytest tests/test_cli.py -x -q` | W0 | pending |
| 09-02-01 | 02 | 2 | TUI-01..06 | unit | `pytest tests/test_tui.py -x -q` | W0 | pending |
| 09-02-02 | 02 | 2 | TUI-06 | unit | `pytest tests/test_tui.py -k handoff -x -q` | W0 | pending |

*Status: pending / green / red / flaky*

---

## Wave 0 Requirements

- [ ] `tests/test_cli.py` — stubs for all CLI subcommands
- [ ] `tests/test_tui.py` — stubs for TUI loading screen, menu, handoff
- [ ] `typer>=0.12` — add to pyproject.toml dependencies

*Existing test infrastructure (pytest, conftest) covers framework needs.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Feature ticker scrolling animation | TUI-03 | Visual animation quality | Run `ocsd`, watch loading screen, verify feature text scrolls |
| Arrow-key menu navigation | TUI-04 | Interactive keyboard input | Navigate menu with arrow keys, verify highlight moves |
| Terminal minimize during overlay | TUI-06 | Platform-specific window management | Launch record/run, verify terminal minimizes |
| V2 features greyed out | TUI-05 | Visual styling quality | Check menu shows greyed items with "Feature inbound" |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 10s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
