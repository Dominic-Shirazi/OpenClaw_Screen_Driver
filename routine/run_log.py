"""Run directory management, screenshot annotation, and self-cleaning.

Creates per-run directories under ``routine_dir/runs/``, saves annotated
screenshots and result JSON files, and prunes old runs to keep disk
usage bounded.
"""

from __future__ import annotations

import json
import logging
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from core.config import get_config

logger = logging.getLogger(__name__)


def create_run_dir(routine_dir: Path) -> Path:
    """Create a timestamped run directory for a routine execution.

    Args:
        routine_dir: Root directory of the routine.

    Returns:
        Path to the newly created run directory.
    """
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
    run_dir = routine_dir / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Created run directory: %s", run_dir)
    return run_dir


def save_run_result(run_dir: Path, result: dict[str, Any]) -> None:
    """Write the run result dict to ``result.json`` in the run directory.

    Args:
        run_dir: Path to the run directory.
        result: Dictionary with run metadata. Expected keys include
            ``run_id``, ``routine_name``, ``status``, ``started_at``,
            ``completed_at``, ``duration_ms``, ``steps_completed``,
            ``total_steps``, ``failure_step``, ``failure_reason``.
    """
    result_path = run_dir / "result.json"
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    logger.debug("Saved run result to %s", result_path)


def annotate_screenshot(
    img: np.ndarray,
    bboxes: list[dict[str, Any]],
    label: str = "",
) -> np.ndarray:
    """Draw bounding boxes and an optional label onto a screenshot copy.

    Args:
        img: BGR image array to annotate.
        bboxes: List of dicts with keys ``x``, ``y``, ``w``, ``h`` and
            optional ``color`` (BGR tuple, default green) and ``label``.
        label: Overall label drawn at the top-left corner.

    Returns:
        Annotated copy of the image (original is not modified).
    """
    annotated = img.copy()

    for bbox in bboxes:
        x = bbox.get("x", 0)
        y = bbox.get("y", 0)
        w = bbox.get("w", 0)
        h = bbox.get("h", 0)
        color = bbox.get("color", (0, 255, 0))
        box_label = bbox.get("label", "")

        cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
        if box_label:
            cv2.putText(
                annotated,
                box_label,
                (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
            )

    if label:
        # Dark background for readability
        text_size, _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1
        )
        cv2.rectangle(
            annotated,
            (5, 5),
            (15 + text_size[0], 30 + text_size[1]),
            (0, 0, 0),
            cv2.FILLED,
        )
        cv2.putText(
            annotated,
            label,
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
        )

    return annotated


def save_annotated_screenshot(
    run_dir: Path,
    img: np.ndarray,
    name: str,
) -> Path:
    """Save an image to the run directory as a PNG file.

    Args:
        run_dir: Path to the run directory.
        img: BGR image array.
        name: Base name (without extension) for the output file.

    Returns:
        Path to the saved PNG file.
    """
    out_path = run_dir / f"{name}.png"
    cv2.imwrite(str(out_path), img)
    logger.debug("Saved annotated screenshot: %s", out_path)
    return out_path


def prune_old_runs(routine_dir: Path) -> int:
    """Delete old run directories to keep disk usage bounded.

    Reads retention limits from the ``replay`` config section:
    - ``keep_successful_runs`` (default 5)
    - ``keep_failed_runs`` (default 10)

    Runs without a ``result.json`` are treated as failed.

    Args:
        routine_dir: Root directory of the routine.

    Returns:
        Number of run directories deleted.
    """
    config = get_config().get("replay", {})
    keep_success = config.get("keep_successful_runs", 5)
    keep_failed = config.get("keep_failed_runs", 10)

    runs_dir = routine_dir / "runs"
    if not runs_dir.exists():
        return 0

    # Collect all run subdirs sorted by name (oldest first)
    all_runs = sorted(
        [d for d in runs_dir.iterdir() if d.is_dir()],
        key=lambda p: p.name,
    )

    successes: list[Path] = []
    failures: list[Path] = []

    for run_path in all_runs:
        result_file = run_path / "result.json"
        status = "failed"
        if result_file.exists():
            try:
                with open(result_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                status = data.get("status", "failed")
            except (json.JSONDecodeError, OSError):
                status = "failed"

        if status == "success":
            successes.append(run_path)
        else:
            failures.append(run_path)

    deleted = 0

    # Delete excess successful runs (oldest first)
    if len(successes) > keep_success:
        to_delete = successes[: len(successes) - keep_success]
        for path in to_delete:
            logger.info("Pruning old successful run: %s", path.name)
            shutil.rmtree(path)
            deleted += 1

    # Delete excess failed runs (oldest first)
    if len(failures) > keep_failed:
        to_delete = failures[: len(failures) - keep_failed]
        for path in to_delete:
            logger.info("Pruning old failed run: %s", path.name)
            shutil.rmtree(path)
            deleted += 1

    if deleted:
        logger.info("Pruned %d old run(s) from %s", deleted, routine_dir.name)

    return deleted


def setup_run_logger(run_dir: Path) -> logging.FileHandler:
    """Create a file handler that logs to ``run.log`` in the run directory.

    Args:
        run_dir: Path to the run directory.

    Returns:
        Configured FileHandler. Caller is responsible for adding it
        to the root logger and removing it after the run completes.
    """
    handler = logging.FileHandler(run_dir / "run.log", encoding="utf-8")
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    return handler
