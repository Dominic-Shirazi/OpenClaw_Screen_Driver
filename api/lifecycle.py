"""Daemon thread server lifecycle for OCSD API.

Starts the FastAPI/uvicorn server as a daemon thread so it can run
alongside the Qt event loop without conflicting with signal handlers.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

import uvicorn

from core.config import get_config

logger = logging.getLogger(__name__)


class _NoSignalServer(uvicorn.Server):
    """Uvicorn server that does not install signal handlers.

    Qt owns the signal handlers in OCSD; uvicorn must not override them
    or the application will fail to shut down cleanly.
    """

    def install_signal_handlers(self) -> None:
        """Skip signal handler installation -- Qt owns signals."""
        pass


def start_api_daemon(
    app: Any,
    host: str | None = None,
    port: int | None = None,
) -> threading.Thread:
    """Start the API server as a daemon thread.

    Args:
        app: The FastAPI/ASGI application to serve.
        host: Override host binding. Defaults to config value.
        port: Override port. Defaults to config value.

    Returns:
        The daemon thread running the server.
    """
    config = get_config()
    api_cfg = config.get("api", {})
    h = host or api_cfg.get("host", "127.0.0.1")
    p = port or api_cfg.get("port", 8420)

    uv_config = uvicorn.Config(app, host=h, port=p, log_level="info")
    server = _NoSignalServer(uv_config)

    thread = threading.Thread(target=server.run, daemon=True, name="ocsd-api")
    thread.start()
    logger.info("API daemon started on %s:%d", h, p)
    return thread
