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
