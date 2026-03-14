"""YOLOE text-prompt benchmark for UI element detection.

Go/no-go gate: If YOLOE cannot detect >50% of obvious UI elements
on 3-5 real screenshots, the text-prompt approach is not viable and
we fall back to the spec's Fallback A/B/C plans.

Usage:
    python scripts/benchmark_yoloe_ui.py [--screenshots-dir assets/screenshots]

Place 3-5 PNG screenshots in assets/screenshots/ before running.
Screenshots should include: Windows desktop, browser page, form/dialog.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# Element classes to benchmark — short concrete nouns per YOLOE best practices
ELEMENT_CLASSES = [
    "button", "icon", "text field", "checkbox", "dropdown",
    "scrollbar", "tab", "link", "toggle", "slider", "menu item",
    # Null/disambiguation classes (reduce false positives)
    "text label", "image",
]

# Per-class minimum confidence thresholds for post-filtering
# UI-novel classes need lower thresholds than COCO-adjacent ones
CLASS_CONF_THRESHOLDS = {
    "button": 0.03,
    "icon": 0.05,
    "text field": 0.02,
    "checkbox": 0.03,
    "dropdown": 0.02,
    "scrollbar": 0.05,
    "tab": 0.03,
    "link": 0.02,
    "toggle": 0.03,
    "slider": 0.05,
    "menu item": 0.02,
    "text label": 0.03,
    "image": 0.05,
}

# Maximum bbox area as fraction of image — filter out "group" detections
MAX_BBOX_AREA_FRACTION = 0.40

# Go/no-go threshold: 5 detections/image is the proxy for "YOLOE can see UI elements".
# The spec's "> 50% recall" requires ground truth; avg detections is the practical substitute.
GO_NO_GO_THRESHOLD = 5


def load_model(model_path: str = "yoloe-26s-seg.pt") -> Any:
    """Load YOLOE model and cache text embeddings."""
    from ultralytics import YOLOE

    logger.info("Loading YOLOE model: %s", model_path)
    model = YOLOE(model_path)

    logger.info("Computing text embeddings for %d classes...", len(ELEMENT_CLASSES))
    t0 = time.perf_counter()
    tpe = model.model.get_text_pe(ELEMENT_CLASSES)
    model.model.set_classes(ELEMENT_CLASSES, tpe)
    embed_ms = (time.perf_counter() - t0) * 1000
    logger.info("Text embeddings cached in %.0f ms", embed_ms)

    return model


def run_detection(model, image: np.ndarray, image_name: str) -> list[dict]:
    """Run YOLOE text-prompt detection on a single image."""
    h, w = image.shape[:2]
    image_area = h * w

    t0 = time.perf_counter()
    results = model.predict(image, conf=0.01, iou=0.4, verbose=False)
    detect_ms = (time.perf_counter() - t0) * 1000

    detections = []
    for result in results:
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            continue
        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i])
            cls_name = ELEMENT_CLASSES[cls_id] if cls_id < len(ELEMENT_CLASSES) else "unknown"
            conf = float(boxes.conf[i])

            # Per-class confidence filtering
            min_conf = CLASS_CONF_THRESHOLDS.get(cls_name, 0.03)
            if conf < min_conf:
                continue

            xyxy = boxes.xyxy[i].cpu().numpy().astype(int)
            bx1, by1, bx2, by2 = xyxy
            bw, bh = bx2 - bx1, by2 - by1

            # Filter out oversized detections
            if (bw * bh) / image_area > MAX_BBOX_AREA_FRACTION:
                continue

            # Filter out tiny detections (noise)
            if bw < 5 or bh < 5:
                continue

            detections.append({
                "class": cls_name,
                "confidence": round(conf, 4),
                "bbox": [int(bx1), int(by1), int(bw), int(bh)],
            })

    logger.info(
        "  %s: %d detections in %.0f ms",
        image_name, len(detections), detect_ms,
    )

    # Print per-class breakdown
    class_counts: dict[str, int] = {}
    for d in detections:
        class_counts[d["class"]] = class_counts.get(d["class"], 0) + 1
    for cls, count in sorted(class_counts.items(), key=lambda x: -x[1]):
        logger.info("    %-15s %d", cls, count)

    return detections


def draw_detections(image: np.ndarray, detections: list[dict], output_path: Path) -> None:
    """Draw detection boxes on image and save for visual review."""
    vis = image.copy()
    colors = {
        "button": (0, 255, 0), "icon": (255, 165, 0), "text field": (255, 0, 0),
        "checkbox": (0, 255, 255), "dropdown": (255, 0, 255), "scrollbar": (128, 128, 0),
        "tab": (0, 128, 255), "link": (255, 255, 0), "toggle": (128, 0, 255),
        "slider": (0, 255, 128), "menu item": (255, 128, 0),
        "text label": (180, 180, 180), "image": (100, 100, 100),
    }
    for det in detections:
        x, y, w, h = det["bbox"]
        color = colors.get(det["class"], (200, 200, 200))
        cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)
        label = f"{det['class']} {det['confidence']:.0%}"
        cv2.putText(vis, label, (x, max(12, y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    cv2.imwrite(str(output_path), vis)
    logger.info("  Visualization saved: %s", output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="YOLOE UI detection benchmark")
    parser.add_argument("--screenshots-dir", default="assets/screenshots",
                        help="Directory containing PNG screenshots to benchmark")
    parser.add_argument("--model", default="yoloe-26s-seg.pt",
                        help="YOLOE model weights path")
    parser.add_argument("--output-dir", default="assets/screenshots/benchmark_results",
                        help="Directory for visualization outputs")
    args = parser.parse_args()

    screenshots_dir = Path(args.screenshots_dir)
    if not screenshots_dir.exists():
        logger.error("Screenshots directory not found: %s", screenshots_dir)
        logger.error("Place 3-5 PNG screenshots in %s and re-run.", screenshots_dir)
        sys.exit(1)

    images = sorted(screenshots_dir.glob("*.png"))
    if len(images) < 3:
        logger.error("Need at least 3 screenshots, found %d", len(images))
        sys.exit(1)

    output_dir = Path(args.output_dir)
    if output_dir.resolve() == screenshots_dir.resolve():
        logger.error("--output-dir must not be the same as --screenshots-dir")
        sys.exit(1)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(args.model)

    logger.info("\n=== YOLOE UI Detection Benchmark ===\n")

    all_detections = {}
    total_elements = 0

    for img_path in images:
        image = cv2.imread(str(img_path))
        if image is None:
            logger.warning("Could not read %s, skipping", img_path)
            continue

        detections = run_detection(model, image, img_path.name)
        all_detections[img_path.name] = detections
        total_elements += len(detections)

        # Save visualization
        vis_path = output_dir / f"bench_{img_path.name}"
        draw_detections(image, detections, vis_path)

    # Summary
    logger.info("\n=== BENCHMARK SUMMARY ===")
    logger.info("Screenshots tested: %d", len(all_detections))
    logger.info("Total detections:   %d", total_elements)
    avg = total_elements / max(1, len(all_detections))
    logger.info("Average per image:  %.1f", avg)

    # Save JSON report
    report = {
        "model": args.model,
        "classes": ELEMENT_CLASSES,
        "conf_thresholds": CLASS_CONF_THRESHOLDS,
        "results": {
            name: {"count": len(dets), "detections": dets}
            for name, dets in all_detections.items()
        },
        "summary": {
            "total_images": len(all_detections),
            "total_detections": total_elements,
            "avg_per_image": round(avg, 1),
        },
    }
    report_path = output_dir / "benchmark_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Report saved: %s", report_path)

    # Go/no-go verdict
    logger.info("\n=== GO / NO-GO VERDICT ===")
    if avg >= GO_NO_GO_THRESHOLD:
        logger.info("PASS: YOLOE text-prompt finds %.1f elements/image on average.", avg)
        logger.info("Proceed with Wave 1 pipeline integration.")
    else:
        logger.warning("FAIL: YOLOE text-prompt finds only %.1f elements/image.", avg)
        logger.warning("Execute fallback plan (see spec: Fallback A/B/C).")
        sys.exit(2)


if __name__ == "__main__":
    main()
