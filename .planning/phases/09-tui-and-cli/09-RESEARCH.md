# Phase 9: TUI and CLI - Research

**Researched:** 2026-03-20
**Domain:** Terminal UI (Rich) + CLI framework (Typer) + Qt event loop handoff
**Confidence:** HIGH

## Summary

Phase 9 replaces the existing `main.py` argparse-based entry point and `recorder/tui.py` Rich menu with a Typer-based CLI (`ocsd` command) and a Rich TUI loading screen that serves as the model initialization gate. The existing codebase already has all the backend functions wired up: `routine/runner.py:run_routine()`, `routine/management.py:fork_routine()/delete_routine()/inspect_routine()`, `routine/discovery.py:list_routines()`, `routine/update_session.py:UpdateSession`, and `recorder/record_session.py:RecordSession`. The CLI layer is pure wiring -- no new business logic.

The critical architectural constraint is the sequential gate pattern: the Rich TUI event loop must complete and be fully destroyed before `QApplication()` is created. Two event loops must never coexist. This is achievable because Rich uses synchronous rendering (no event loop) -- `rich.progress.Progress` and `rich.live.Live` are context managers that run in the main thread and exit cleanly.

**Primary recommendation:** Use Typer 0.24+ with Rich 14.3+ for the CLI/TUI stack. Structure as a single `cli/` package with `app.py` (Typer app), `tui.py` (loading screen + menu), and `output.py` (JSON/table output helpers). Keep the TUI as a synchronous Rich Live display, not a Textual app.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **Typer** for CLI framework (built on Click, type hints for args, auto-generated help)
- **Always show TUI first**: Even CLI subcommands show the TUI loading screen first as initialization gate. Only API-triggered runs (Phase 10) skip TUI
- **Both name and path**: `ocsd run "My Routine"` looks up by name, `ocsd run path/to/routine.json` runs from any path. Auto-detect by checking if argument is a file path
- **--speed flag**: `ocsd run "Name" --speed 0` for instant, `--speed 1.0` normal, `--speed 5.0` slow
- **--json flag**: All commands support `--json` for machine-readable output
- **--param flag**: `ocsd run "Name" --param search_term="news"` passes variable values without interactive prompting
- **Rich error panels**: Red-bordered Rich panel with error title, message, and fix suggestion
- **Hub search stub**: `ocsd hub search` prints "Hub coming soon in V2"
- **Pre-run variable collection**: Scan all steps for variable inputs, prompt user for ALL values upfront with hints, then execute uninterrupted
- **Feature ticker from data file**: `features.yml` or `features.md` bundled with package, each entry has text + status (shipped/coming_soon)
- **Arrow-key navigable menu**: Record / Run / Update / Fork / List with highlight, V2+ items greyed out with "Feature inbound"
- **Two-step action flow**: Menu shows actions first, then searchable routine list for Run/Update/Fork
- **Searchable routine list**: Type to filter in real-time, arrow keys to navigate
- **Terminal minimizes**: During overlay operation, terminal auto-minimizes. Restore with summary after
- **Sequential gate**: TUI Rich event loop completes fully and is destroyed before QApplication() is created

### Claude's Discretion
- Exact Typer app structure (single file vs subcommand modules)
- How to implement terminal minimize/restore cross-platform
- Feature ticker animation speed and visual style
- How to detect whether an argument is a routine name or file path
- Rich table column formatting for `ocsd list` and `ocsd inspect`
- How to implement searchable routine list (Rich Live vs custom)
- Model loading progress bar implementation (real progress vs estimated)
- How to structure the features.yml data file

### Deferred Ideas (OUT OF SCOPE)
- Inline variable resolution (using output from step N as input to step N+1) -- V2
- Remote Routine Hub with network sync -- V2
- Voice input during recording (faster-whisper) -- V2
- GUI launcher (Raycast-style) -- V2
- Bundled demo routines -- user creates when making first routines
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| TUI-01 | Rich TUI as home base -- routine browser with searchable list | Rich Live + keyboard input for real-time filtering; `list_routines()` provides data |
| TUI-02 | TUI shows model loading status | Rich Progress with multiple tasks (OmniParser, CLIP, VLM check) |
| TUI-03 | TUI loading screen with feature ticker scrolling | Rich Live display with scrolling text from `features.yml` data file |
| TUI-04 | TUI menu: Record / Run / Update / Fork | Rich-based arrow-key menu with highlight; dispatches to CLI commands |
| TUI-05 | Greyed-out V2+ features with "Feature inbound" labels | Rich Text markup with dim styling on disabled items |
| TUI-06 | TUI runs before Qt event loop (sequential gate) | Rich is synchronous -- no event loop conflict. Function returns before Qt starts |
| CLI-01 | `ocsd` launches TUI (no arguments) | Typer callback with `invoke_without_command=True` |
| CLI-02 | `ocsd record "Name"` | Typer command wrapping `RecordSession` via `cmd_record` flow |
| CLI-03 | `ocsd run "Name"` or `ocsd run path/to/routine.json` | Typer command with name-or-path detection, wrapping `run_routine()` |
| CLI-04 | `ocsd list` (human table) and `ocsd list --json` | Typer command wrapping `list_routines()` with Rich table or JSON output |
| CLI-05 | `ocsd inspect "Name"` | Typer command wrapping `inspect_routine()` (already has `as_json` param) |
| CLI-06 | `ocsd update "Name"` | Typer command wrapping `UpdateSession` |
| CLI-07 | `ocsd fork "Name" "NewName"` | Typer command wrapping `fork_routine()` |
| CLI-08 | `ocsd hub search "query"` | Typer command with stub message |
| CLI-09 | All commands support `--json` flag | Typer `Option` with callback or per-command flag |
</phase_requirements>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| typer | >=0.24 | CLI framework with subcommands | Type-hint-based, auto help/completion, created by FastAPI team. Already decided in CONTEXT.md |
| rich | >=13.0 | Terminal rendering (tables, panels, progress, Live) | Already a dependency (tui optional group). Used by existing `management.py` |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| pyyaml | >=6.0 | Parse features.yml data file | Already a project dependency |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Typer | Click | Typer wraps Click with type hints; Click is more verbose but equally capable |
| Rich Live menu | Textual | Textual is a full TUI framework with event loop -- conflicts with the "no concurrent event loops" constraint. Rich Live is synchronous |
| Rich prompt for searchable list | prompt_toolkit | Extra dependency. Rich Live with keyboard polling achieves the same result |

**Installation:**
```bash
pip install typer[all]>=0.24
# rich is already installed via tui optional group
```

**pyproject.toml changes:**
```toml
# Move rich from optional to core deps, add typer
dependencies = [
    # ... existing ...
    "rich>=13.0",
    "typer>=0.24",
]

[project.scripts]
ocsd = "cli.app:main"
```

## Architecture Patterns

### Recommended Project Structure
```
cli/
    __init__.py
    app.py           # Typer app definition, all @app.command() entries
    tui.py           # Loading screen + interactive menu (Rich Live)
    output.py        # Shared output helpers (json_or_table, error_panel)
    features.yml     # Feature ticker data file
    _minimize.py     # Cross-platform terminal minimize/restore
```

### Pattern 1: Typer App with Callback for No-Args TUI
**What:** Use Typer's `invoke_without_command=True` callback to launch TUI when no subcommand is given.
**When to use:** CLI-01 -- `ocsd` with no arguments opens TUI.
**Example:**
```python
# Source: Typer docs - typer command callback
import typer

app = typer.Typer(
    name="ocsd",
    help="OpenClaw Screen Driver -- AI-powered screen automation",
    no_args_is_help=False,
    rich_markup_mode="rich",
)

@app.callback(invoke_without_command=True)
def main_callback(ctx: typer.Context) -> None:
    """Launch TUI if no subcommand given."""
    if ctx.invoked_subcommand is None:
        from cli.tui import run_tui
        run_tui()
```

### Pattern 2: Name-or-Path Detection
**What:** Auto-detect whether a CLI argument is a routine name (lookup in `~/.ocsd/routines/`) or a file path.
**When to use:** CLI-03 -- `ocsd run "My Routine"` vs `ocsd run ./path/to/routine.json`.
**Example:**
```python
from pathlib import Path
from routine.discovery import get_routine_dir

def resolve_routine_dir(name_or_path: str) -> Path:
    """Resolve a routine name or path to its directory."""
    candidate = Path(name_or_path)
    # If it looks like a path (has separator or extension), try as path
    if candidate.exists() and candidate.is_dir():
        return candidate
    if candidate.exists() and candidate.is_file():
        return candidate.parent  # routine.json -> parent dir
    # Otherwise look up by name
    routine_dir = get_routine_dir() / name_or_path
    if routine_dir.exists():
        return routine_dir
    raise FileNotFoundError(f"Routine not found: {name_or_path}")
```

### Pattern 3: Sequential Gate (TUI before Qt)
**What:** TUI loading screen runs as synchronous Rich context manager, returns control, then Qt starts.
**When to use:** TUI-06 -- ensuring two event loops never coexist.
**Example:**
```python
def launch_command(action: str, routine_dir: Path | None = None) -> int:
    """Run TUI loading screen, then dispatch to Qt-based action."""
    # Phase 1: Rich TUI (synchronous, no event loop)
    from cli.tui import show_loading_screen
    show_loading_screen()  # Returns when loading is complete

    # Phase 2: Qt application (starts its own event loop)
    from PyQt6.QtWidgets import QApplication
    import sys
    app = QApplication.instance() or QApplication(sys.argv)
    # ... dispatch to record/run/update ...
    return app.exec()
```

### Pattern 4: Pre-Run Variable Collection
**What:** Scan routine steps for `input_spec.type == "variable"`, prompt user for all values upfront, inject into steps before replay.
**When to use:** Before `run_routine()` is called, when `--param` flags don't cover all variables.
**Example:**
```python
from rich.prompt import Prompt

def collect_variables(
    routine: Routine,
    provided_params: dict[str, str],
) -> dict[str, str]:
    """Scan steps for variables and prompt for missing values."""
    needed: dict[str, str] = {}  # var_name -> hint
    for step in routine.steps:
        spec = step.get("input_spec", {})
        if spec.get("type") == "variable":
            var_name = spec["value"].strip("{}")
            hint = spec.get("hint", f"Enter {var_name}")
            if var_name not in provided_params:
                needed[var_name] = hint

    collected = dict(provided_params)
    for var_name, hint in needed.items():
        value = Prompt.ask(f"[bold]{hint}[/]")
        collected[var_name] = value
    return collected
```

### Pattern 5: --json Flag with Dual Output
**What:** Every command checks a `--json` flag and switches between Rich table and JSON output.
**When to use:** CLI-09 -- all commands support `--json`.
**Example:**
```python
import json
from rich.console import Console
from rich.table import Table

def output_routines(routines: list[RoutineInfo], as_json: bool) -> None:
    """Output routine list as table or JSON."""
    if as_json:
        data = [{"name": r.name, "path": str(r.path), "version": r.schema_version}
                for r in routines]
        print(json.dumps(data, indent=2))
        return

    console = Console()
    table = Table(title="Routines", border_style="blue")
    table.add_column("Name", style="bold")
    table.add_column("Version", style="dim")
    table.add_column("Path", style="dim")
    for r in routines:
        table.add_row(r.name, r.schema_version, str(r.path))
    console.print(table)
```

### Pattern 6: Terminal Minimize/Restore (Cross-Platform)
**What:** Minimize the terminal window when overlay launches, restore when it closes.
**When to use:** TUI-to-Qt handoff.
**Example:**
```python
import sys

def minimize_terminal() -> None:
    """Minimize the terminal window (best-effort, platform-specific)."""
    if sys.platform == "win32":
        import ctypes
        hwnd = ctypes.windll.kernel32.GetConsoleWindow()
        if hwnd:
            ctypes.windll.user32.ShowWindow(hwnd, 6)  # SW_MINIMIZE
    # Linux/macOS: no reliable cross-platform way; skip silently

def restore_terminal() -> None:
    """Restore the terminal window."""
    if sys.platform == "win32":
        import ctypes
        hwnd = ctypes.windll.kernel32.GetConsoleWindow()
        if hwnd:
            ctypes.windll.user32.ShowWindow(hwnd, 9)  # SW_RESTORE
```

### Anti-Patterns to Avoid
- **Textual for TUI**: Textual has its own async event loop that conflicts with Qt's event loop. Use Rich's synchronous `Live` display instead.
- **Concurrent Rich + Qt**: Never have Rich `Live` context open while `QApplication` is running. Rich `Live` takes over terminal output and would conflict.
- **Global --json via Typer callback**: Don't try to make `--json` a global option via callback state. Instead, add it as a parameter to each command. Simpler and more explicit.
- **Keeping TUI alive during overlay**: The TUI must exit completely. Don't try to keep a "background" TUI running while Qt overlay is active.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| CLI argument parsing | Custom argparse wrappers | Typer `@app.command()` with type hints | Auto help, completion, validation |
| Progress bars | Custom terminal animation | `rich.progress.Progress` | Handles terminal width, ETA, multi-task |
| Table formatting | String concatenation with padding | `rich.table.Table` | Auto-column width, borders, styles |
| JSON pretty-printing | `json.dumps` + manual formatting | `rich.json.JSON` for display, `json.dumps` for `--json` | Rich syntax-highlights JSON |
| Error display | `print("Error: ...")` | `rich.panel.Panel` with red border | Consistent, scannable error format |
| Arrow-key menu | Raw terminal input handling | Rich `Live` + `msvcrt`/`sys.stdin` keypress | Avoid reimplementing terminal input |

**Key insight:** Rich already provides every rendering primitive needed. The TUI is just orchestrating Rich widgets in the right sequence.

## Common Pitfalls

### Pitfall 1: Qt Event Loop Created Before TUI Exits
**What goes wrong:** `QApplication()` is instantiated while Rich `Live` context is still active, causing terminal rendering corruption.
**Why it happens:** Developer imports Qt modules at the top of the CLI file, or creates `QApplication` in a shared setup function.
**How to avoid:** Lazy-import all Qt modules inside the function that needs them, AFTER TUI has returned. The existing codebase already follows this pattern (`cmd_record` lazy-imports `QApplication`).
**Warning signs:** Terminal output garbled after overlay closes; Python crash on exit.

### Pitfall 2: Typer Eating Exceptions
**What goes wrong:** Typer/Click catches exceptions and prints a generic error instead of the Rich error panel.
**Why it happens:** Typer wraps commands in try/except by default.
**How to avoid:** Use `app = typer.Typer(pretty_exceptions_enable=False)` or catch exceptions inside the command function and render Rich panels before calling `raise typer.Exit(code=1)`.
**Warning signs:** Seeing Click-style error output instead of Rich panels.

### Pitfall 3: --param Parsing with Equals Signs
**What goes wrong:** `--param search_term="hello world"` gets split incorrectly.
**Why it happens:** Shell quoting interacts with Typer's argument parsing.
**How to avoid:** Use `typer.Option(parser=parse_param)` or accept `--param` as a `list[str]` and parse `key=value` manually. Each `--param` invocation is one key=value pair.
**Warning signs:** Variable values truncated at spaces or equals signs.

### Pitfall 4: Searchable List Blocking Input
**What goes wrong:** Rich `Live` display doesn't receive keystrokes because `input()` blocks.
**Why it happens:** `Live` refreshes the display but doesn't handle keyboard input natively.
**How to avoid:** Use a polling loop: on Windows use `msvcrt.kbhit()` + `msvcrt.getwch()`, on Unix use `select.select()` on stdin with raw terminal mode. Update the `Live` renderable on each keypress.
**Warning signs:** Menu appears but doesn't respond to typing; terminal hangs.

### Pitfall 5: run_routine Missing execution_params
**What goes wrong:** Pre-run variable values are collected but never passed to the step executor.
**Why it happens:** The Phase 7 `run_routine()` signature doesn't have an `execution_params` parameter. The legacy `mapper/runner.py` does, but the new routine runner doesn't.
**How to avoid:** Either (a) inject resolved values into `step["text_to_type"]` before calling `run_routine()`, or (b) extend `run_routine()` to accept `execution_params` and resolve variables internally. Option (a) is simpler and doesn't modify the runner.
**Warning signs:** Variable steps type empty strings during replay.

### Pitfall 6: Console Window Handle on Non-Console Launch
**What goes wrong:** `GetConsoleWindow()` returns 0 when launched from a non-console context (IDE, subprocess).
**Why it happens:** Not all terminal emulators have a Win32 console window.
**How to avoid:** Check return value before calling `ShowWindow`. Make minimize/restore best-effort with silent fallback.
**Warning signs:** ctypes error or crash when launched from VS Code terminal.

## Code Examples

### Typer App Structure
```python
# cli/app.py
from __future__ import annotations

import typer

app = typer.Typer(
    name="ocsd",
    help="OpenClaw Screen Driver",
    no_args_is_help=False,
    rich_markup_mode="rich",
    pretty_exceptions_enable=False,
)

@app.callback(invoke_without_command=True)
def main_callback(ctx: typer.Context) -> None:
    if ctx.invoked_subcommand is None:
        from cli.tui import run_tui
        run_tui()

@app.command()
def record(name: str = typer.Argument(..., help="Routine name")) -> None:
    """Record a new routine."""
    ...

@app.command()
def run(
    name: str = typer.Argument(..., help="Routine name or path"),
    speed: float = typer.Option(1.0, help="Execution speed multiplier"),
    json_output: bool = typer.Option(False, "--json", help="JSON output"),
    param: list[str] = typer.Option([], "--param", help="Variable values (key=value)"),
) -> None:
    """Run a routine."""
    ...

@app.command()
def list_(json_output: bool = typer.Option(False, "--json")) -> None:
    """List all routines."""
    ...

# ... more commands ...

def main() -> None:
    app()
```

### Rich Loading Screen with Progress
```python
# cli/tui.py
from rich.console import Console
from rich.live import Live
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.panel import Panel
from rich.text import Text

def show_loading_screen() -> None:
    """Display model loading progress with feature ticker."""
    console = Console()

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        console=console,
    )

    omni_task = progress.add_task("OmniParser", total=1)
    clip_task = progress.add_task("CLIP embeddings", total=1)
    vlm_task = progress.add_task("VLM connectivity", total=1)

    # Show progress (synchronous -- blocks until done)
    with Live(progress, console=console, refresh_per_second=10):
        # Load models sequentially
        _load_omniparser()
        progress.update(omni_task, completed=1)
        _load_clip()
        progress.update(clip_task, completed=1)
        _check_vlm()
        progress.update(vlm_task, completed=1)
    # Live context exited -- terminal is clean for Qt
```

### Arrow-Key Menu
```python
import sys

def _read_key() -> str:
    """Read a single keypress (cross-platform)."""
    if sys.platform == "win32":
        import msvcrt
        ch = msvcrt.getwch()
        if ch in ('\x00', '\xe0'):  # Special key prefix
            ch2 = msvcrt.getwch()
            if ch2 == 'H': return 'up'
            if ch2 == 'P': return 'down'
            return ''
        if ch == '\r': return 'enter'
        if ch == '\x1b': return 'escape'
        return ch
    else:
        import tty, termios
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            ch = sys.stdin.read(1)
            if ch == '\x1b':
                ch2 = sys.stdin.read(1)
                ch3 = sys.stdin.read(1)
                if ch2 == '[':
                    if ch3 == 'A': return 'up'
                    if ch3 == 'B': return 'down'
                return 'escape'
            if ch == '\r': return 'enter'
            return ch
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
```

### Features YAML Structure
```yaml
# cli/features.yml
features:
  - text: "Record once, replay forever"
    status: shipped
  - text: "AI-powered element detection (5-stage cascade)"
    status: shipped
  - text: "Human-like mouse movement and typing"
    status: shipped
  - text: "Smart bbox refinement with OmniParser"
    status: shipped
  - text: "Update and fork existing routines"
    status: shipped
  - text: "Loop actions with exit conditions"
    status: shipped
  - text: "Voice-recorded routines"
    status: coming_soon
  - text: "Routine Hub marketplace"
    status: coming_soon
  - text: "Cross-routine composition"
    status: coming_soon
```

### Rich Error Panel
```python
from rich.console import Console
from rich.panel import Panel

def show_error(title: str, message: str, fix: str | None = None) -> None:
    """Display a Rich error panel with optional fix suggestion."""
    console = Console(stderr=True)
    body = f"[white]{message}[/white]"
    if fix:
        body += f"\n\n[dim]Fix: {fix}[/dim]"
    console.print(Panel(body, title=f"[bold red]{title}[/]", border_style="red"))
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| argparse for CLI | Typer (type-hint-based) | 2023+ | Auto help, completion, less boilerplate |
| Custom menu with `input()` | Rich Live with keyboard polling | 2024+ | Animated, responsive, arrow-key navigation |
| `recorder/tui.py` (existing) | `cli/tui.py` (new) | This phase | Complete rewrite -- old TUI used numbered prompts |
| `main.py` entry point | `cli/app.py` via `[project.scripts]` | This phase | Clean `ocsd` command with subcommands |

**Deprecated/outdated:**
- `recorder/tui.py`: Replaced entirely by new `cli/tui.py`
- `main.py`: Replaced by `cli/app.py` as entry point. Keep `main.py` for backward compat but redirect to `cli.app:main`

## Open Questions

1. **How does `run_routine()` receive variable values?**
   - What we know: The Phase 7 `run_routine()` reads `step["text_to_type"]` directly. The legacy `mapper/runner.py._resolve_input_text()` has `execution_params` support.
   - What's unclear: Whether to inject resolved values into `step["text_to_type"]` before calling `run_routine()`, or extend the runner signature.
   - Recommendation: Inject into `step["text_to_type"]` in a pre-processing pass before calling `run_routine()`. Simpler, doesn't modify Phase 7 code.

2. **Model loading -- real progress or simulated?**
   - What we know: OmniParser and CLIP loading are I/O-bound with no progress callbacks. VLM is a connectivity check (fast).
   - What's unclear: Whether to show real per-model progress (requires model loading refactor) or estimated progress.
   - Recommendation: Use `total=None` (indeterminate spinner) for model loading, switching to completed when done. Real progress bars would require threading the model loads.

3. **Searchable routine list implementation**
   - What we know: Rich `Live` can render arbitrary content. Keyboard input requires platform-specific polling.
   - What's unclear: Whether the typing experience will feel responsive with Rich `Live` refresh rates.
   - Recommendation: Use Rich `Live` at 10fps refresh with platform-specific `_read_key()`. This is the same pattern used by many Rich-based TUI menus. Falls back gracefully to numbered list if keyboard polling fails.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest 7.0+ |
| Config file | `pyproject.toml [tool.pytest.ini_options]` |
| Quick run command | `pytest tests/test_cli.py -x` |
| Full suite command | `pytest tests/ -x` |

### Phase Requirements to Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| TUI-01 | Searchable routine browser renders list | unit | `pytest tests/test_tui_menu.py::test_routine_browser -x` | No -- Wave 0 |
| TUI-02 | Model loading status shown in progress | unit | `pytest tests/test_tui_menu.py::test_loading_progress -x` | No -- Wave 0 |
| TUI-03 | Feature ticker loads from features.yml | unit | `pytest tests/test_tui_menu.py::test_feature_ticker -x` | No -- Wave 0 |
| TUI-04 | Menu items present and selectable | unit | `pytest tests/test_tui_menu.py::test_menu_items -x` | No -- Wave 0 |
| TUI-05 | V2+ features greyed out | unit | `pytest tests/test_tui_menu.py::test_greyed_features -x` | No -- Wave 0 |
| TUI-06 | TUI exits before Qt starts | integration | `pytest tests/test_tui_menu.py::test_sequential_gate -x` | No -- Wave 0 |
| CLI-01 | `ocsd` with no args launches TUI callback | unit | `pytest tests/test_cli.py::test_no_args_tui -x` | No -- Wave 0 |
| CLI-02 | `ocsd record` dispatches correctly | unit | `pytest tests/test_cli.py::test_record_command -x` | No -- Wave 0 |
| CLI-03 | `ocsd run` resolves name and path | unit | `pytest tests/test_cli.py::test_run_name_and_path -x` | No -- Wave 0 |
| CLI-04 | `ocsd list` outputs table and JSON | unit | `pytest tests/test_cli.py::test_list_table_and_json -x` | No -- Wave 0 |
| CLI-05 | `ocsd inspect` outputs step summary | unit | `pytest tests/test_cli.py::test_inspect_command -x` | No -- Wave 0 |
| CLI-06 | `ocsd update` starts update session | unit | `pytest tests/test_cli.py::test_update_command -x` | No -- Wave 0 |
| CLI-07 | `ocsd fork` creates copy | unit | `pytest tests/test_cli.py::test_fork_command -x` | No -- Wave 0 |
| CLI-08 | `ocsd hub search` shows stub | unit | `pytest tests/test_cli.py::test_hub_stub -x` | No -- Wave 0 |
| CLI-09 | All commands support --json | unit | `pytest tests/test_cli.py::test_json_flag -x` | No -- Wave 0 |

### Sampling Rate
- **Per task commit:** `pytest tests/test_cli.py tests/test_tui_menu.py -x`
- **Per wave merge:** `pytest tests/ -x`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `tests/test_cli.py` -- Typer CLI command tests using `typer.testing.CliRunner`
- [ ] `tests/test_tui_menu.py` -- TUI rendering/interaction tests (mock keyboard input)
- [ ] `pip install typer>=0.24` -- Add to pyproject.toml dependencies

## Sources

### Primary (HIGH confidence)
- [Typer official docs](https://typer.tiangolo.com/) -- commands, callbacks, subcommands, testing
- [Rich official docs](https://rich.readthedocs.io/en/latest/) -- Progress, Live, Table, Panel, JSON
- [Typer PyPI](https://pypi.org/project/typer/) -- version 0.24.1 (Feb 2026)
- [Rich PyPI](https://pypi.org/project/rich/) -- version 14.3.3 (Feb 2026)
- Existing codebase: `routine/runner.py`, `routine/management.py`, `routine/discovery.py`, `recorder/tui.py`, `main.py`

### Secondary (MEDIUM confidence)
- [PyWinCtl](https://github.com/Kalmat/PyWinCtl) -- cross-platform window control (terminal minimize)
- [Typer SubCommands docs](https://typer.tiangolo.com/tutorial/subcommands/) -- nested command groups

### Tertiary (LOW confidence)
- Terminal minimize via `GetConsoleWindow()` -- works for cmd.exe/PowerShell but may not work in all terminal emulators (Windows Terminal, VS Code integrated terminal). Needs validation.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- Typer and Rich are locked decisions with well-documented APIs
- Architecture: HIGH -- sequential gate pattern is well-understood; existing codebase already lazy-imports Qt
- Pitfalls: HIGH -- event loop separation is the main risk, and Rich's synchronous design eliminates it
- Variable collection: MEDIUM -- `run_routine()` needs pre-processing step to inject values; exact mechanism needs implementation

**Research date:** 2026-03-20
**Valid until:** 2026-04-20 (stable libraries, no breaking changes expected)
