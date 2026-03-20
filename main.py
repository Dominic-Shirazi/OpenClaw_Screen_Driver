"""OCSD entry point -- redirects to cli.app:main.

This file exists for backward compatibility. The primary entry
point is now ``cli.app:main`` registered via ``[project.scripts]``
in ``pyproject.toml``.
"""
from __future__ import annotations

import sys


def main() -> int:
    """Redirect to the Typer CLI entry point."""
    from cli.app import main as cli_main
    cli_main()
    return 0


if __name__ == "__main__":
    sys.exit(main())
