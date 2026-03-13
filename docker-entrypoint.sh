#!/bin/bash
set -e

case "$1" in
    api)
        echo "Starting OCSD API server on :8420"
        exec python -m uvicorn api.server:app --host 0.0.0.0 --port 8420
        ;;
    execute)
        shift
        echo "Running OCSD skill execution: $@"
        exec python main.py --execute "$@"
        ;;
    test)
        echo "Running OCSD test suite"
        exec python -m pytest tests/ -v
        ;;
    *)
        # Pass through any other command
        exec "$@"
        ;;
esac
