# OCSD — Gemini CLI Rules

## NON-NEGOTIABLE: Virtual Environment

**ALWAYS activate the project venv before running ANY Python or pip command.**

```bash
# Windows (Git Bash / MSYS2):
source .venv/Scripts/activate

# Windows (CMD):
.venv\Scripts\activate.bat

# Windows (PowerShell):
.venv\Scripts\Activate.ps1

# macOS / Linux:
source .venv/bin/activate
```

- If `.venv/` does not exist, **CREATE IT FIRST**: `python -m venv .venv`
- **NEVER** run `pip install` against the system Python.
- Verify with `which python` — it MUST show `.venv/` in the path.
- After installing anything: `pip freeze > requirements.txt`

## Cross-Platform Awareness

This project runs on **Windows, macOS, and Linux**. Code accordingly:

- Use `pathlib.Path` for all file paths. Never hardcode `\` or `/`.
- Guard platform-specific code with `sys.platform` checks.
- Windows-only modules (`pywinauto`, Win32 ctypes): must fail gracefully on other platforms.
- Test imports before using platform-specific packages.

## Code Standards

- **Type hints** on every function signature.
- **Docstrings** on every public function (Google style).
- Use `from __future__ import annotations` at top of every file.
- Use `logging.getLogger(__name__)`, never `print()` for diagnostics.
- Catch specific exceptions, never bare `except:`.
- Import shared types from `core.types` (Point, Rect, ReplayLog, etc.).
- Import config from `core.config` via `get_config()`.

## Project Structure

```
core/          — sensing & control layer (capture, vision, OCR, embeddings, executor)
recorder/      — recording UI & session (overlay, dialog, element types)
mapper/        — graph construction (NetworkX graph, layers, export)
executor/      — execution engine (pathfinder, runner, validator)
hub/           — skill sharing & security (schema, scanner, manifest)
api/           — FastAPI MCP server
```

## What You Should Know

- **UI Framework:** PyQt6 (not tkinter)
- **VLM:** Qwen2-VL via Ollama (local inference)
- **Embeddings:** CLIP (openai/clip-vit-base-patch32) + FAISS
- **Config:** `config.yaml` loaded by `core/config.py`
- **Voice Input:** Stubbed in Stage 1
- **All images** use BGR format (OpenCV convention) internally

## When Writing Modules

1. Read the relevant section of `Project_blueprint.md` first.
2. Match function signatures exactly as specified in the blueprint.
3. Import types from `core.types`, config from `core.config`.
4. Write ONLY the file you're asked to write. Don't modify other files.
5. Keep the module self-contained — don't add dependencies not in `pyproject.toml`.

## Dependencies

All deps are in `pyproject.toml`. The heavy ones (torch, transformers, faiss-cpu) are installed in the venv. If you need a new dependency, add it to `pyproject.toml` AND install it in the venv.
