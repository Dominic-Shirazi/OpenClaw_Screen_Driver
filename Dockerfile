# OCSD — OpenClaw Screen Driver
# Headless container for API server and CLI replay.
# GUI features (overlay, recording) require a display server and are
# not supported inside this container.
#
# Build:
#   docker build -t ocsd .
#
# Run API server:
#   docker run -p 8420:8420 --env-file .env ocsd api
#
# Run skill replay (headless, dry-run):
#   docker run --env-file .env ocsd execute --dry-run skills/my_skill.json
#
# Connect to external Ollama:
#   docker run -e LITELLM_BASE_URL=http://host.docker.internal:11434 ocsd api

FROM python:3.12-slim AS base

# System deps: Tesseract OCR, OpenCV runtime libs
RUN apt-get update && apt-get install -y --no-install-recommends \
        tesseract-ocr \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps (core only — no GPU/torch in container)
COPY pyproject.toml ./
RUN pip install --no-cache-dir -e ".[api,vlm]"

# Copy application code
COPY . .

# Default config
ENV OCSD_CONFIG_PATH=/app/config.yaml
ENV PYTHONUNBUFFERED=1

EXPOSE 8420

# Entrypoint: dispatch to API server or CLI
COPY docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh
ENTRYPOINT ["/docker-entrypoint.sh"]
CMD ["api"]
