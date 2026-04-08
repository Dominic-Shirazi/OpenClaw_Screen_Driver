# OCSD — Claude Code Rules

## NON-NEGOTIABLE: Environment Isolation

**ALWAYS use the project venv.** Never install packages to system Python.

```bash
# Activate before ANY pip install or python command:
# Windows:  .venv\Scripts\activate
# Mac/Linux: source .venv/bin/activate
```

- If `.venv/` does not exist, **CREATE IT FIRST** before doing anything else.
- If the venv is not activated, **STOP AND ACTIVATE IT** before running pip or python.
- Never run `pip install` without confirming the venv is active (`which python` should show `.venv/`).
- When handing off to Gemini CLI, ensure Gemini's working directory has the venv active or explicitly prefix commands.

## Cross-Platform Awareness

This project targets **Windows, macOS, and Linux**. Every module must account for this:

- **File paths:** Always use `pathlib.Path`, never hardcoded separators.
- **Platform-specific code:** Guard with `sys.platform` checks. Windows-only modules (overlay, accessibility) must have graceful fallbacks or clear "not supported" errors on other platforms.
- **Dependencies:** Some packages are platform-specific:
  - `pywinauto` → Windows only
  - `pyautogui` → works cross-platform but behavior varies
  - Win32 ctypes calls → Windows only, always guard with platform check
- **Line endings:** Use `.gitattributes` to enforce consistent line endings.
- **Tesseract binary:** Different install paths per OS. Config should support this.

## Dependency Management

- All deps go in `pyproject.toml`.
- Heavy deps (torch, transformers, faiss-cpu) should be in optional dependency groups so users can install only what they need.
- Pin major versions for reproducibility.
- Document system-level dependencies (Tesseract, Ollama) in README.

## Code Quality Rules

1. **Type hints on every function signature.** No exceptions.
2. **Docstrings on every public function.** Google style.
3. **Imports:** Use `from __future__ import annotations` for modern type syntax.
4. **Logging:** Use `logging.getLogger(__name__)`, never `print()` for diagnostics.
5. **Error handling:** Catch specific exceptions, never bare `except:`.
6. **Thread safety:** Any module that touches PyAutoGUI, GUI, or shared state must use locks.

## Git Hygiene

- One module per commit during initial build.
- Commit messages: `feat(<module>): description`
- Never commit `.venv/`, `__pycache__/`, `.pyc`, `*.index`, model weights.
- `.gitignore` must be up to date.

## Testing

- Every wave gets a verification test.
- Mock external services (Ollama, Tesseract) in unit tests.
- Integration tests use a local HTML page, not external sites.

## Gemini CLI Handoff Rules

When delegating to Gemini:
- Always specify the full module spec and expected imports.
- Always review output with `git diff` before committing.
- Use `-m flash` for trivial tasks, `-m pro` for research-heavy.
- If Gemini Pro is rate-limited, fall back to flash or write it yourself.
- Gemini should be run from the project root with venv active.

## Architecture Decisions

- **UI Framework:** PyQt6
- **Voice Input:** Stubbed in Stage 1, faster-whisper in Stage 2
- **Graph Library:** NetworkX
- **VLM:** Qwen2-VL via Ollama (local)
- **Embeddings:** CLIP (openai/clip-vit-base-patch32) + FAISS
- **Config:** YAML with Python loader, sensible defaults

## Deployment Considerations

- Docker containerization is planned for distribution.
- macOS users (Mac Mini fleet) are a primary audience.
- GPU support is optional — CPU fallback must always work.
- Ollama must be running externally; OCSD connects to it, doesn't manage it.

## AgentFiles Orchestrator Protocol

This project uses AgentFiles. If `/agentfiles` has been invoked and `project.md` exists,
the following rules are **NON-NEGOTIABLE**:

### Orchestrator Boundaries (YOU, the main conversation)

1. **NEVER Read .py/.yaml/.toml files directly.** You don't need to see the code.
   Dispatch a file-agent. The only files you may read/edit are:
   `project.md`, `.agentfiles/**`, `CLAUDE.md`, `.claude/**`, and memory files.
2. **NEVER Edit or Write code files.** All code changes go through file-agents.
3. **NEVER run `git diff` on code files.** File-agents self-review their own diffs.
4. **You decide WHAT and WHY. The agent decides HOW.** Your briefing describes the
   desired outcome and the reason. The agent chooses the implementation. If you
   catch yourself specifying line numbers or code snippets, you've crossed the line.
5. **Codegraph MCP is always allowed** for structural queries (who calls what,
   dependency graphs, function signatures). It provides architecture, not file content.

### File-Agent Boundaries (subagents YOU dispatch)

1. **Only read your assigned file(s).** If you need info about another file,
   use codegraph MCP for structural queries, or report back to the orchestrator
   asking them to query that file's agent.
2. **NEVER edit files outside your assignment.** If you see a bug in another file,
   report it — don't fix it.
3. **The orchestrator's briefing is a HYPOTHESIS, not truth.** Your file is ground
   truth. Verify every claim against what you actually see. If reality doesn't match
   the briefing, report the discrepancy — don't force the fix.
4. **Claims about other files are UNTRUSTED.** If the orchestrator says "vision.py
   returns X" — verify via codegraph MCP or report back that you need confirmation.
   Do NOT write code that assumes unverified claims about other files' behavior.
5. **Report what you SEE, not what you were told to see.** If the orchestrator says
   "fix the NameError on line 179" but the real issue is on line 203, say so.
6. **Push back when the requested change conflicts with your file's annotations
   or would break its internal logic.** You are the expert. The orchestrator
   coordinates — you protect your file's integrity.
