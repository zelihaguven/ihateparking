"""Lightweight vision-assist analysis for a fixed-camera driving/parking clip.

This v1 intentionally reports visual cues rather than pretending to know the
real steering angle or vehicle CAN state. It detects orange boundary cones and
turns them into human-in-the-loop guidance events.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def _instruction(centroid_x: float | None, area_ratio: float, blob_count: int) -> tuple[str, str]:
    if area_ratio > 0.035:
        return "STOP — boundary is too close", "large orange boundary response"
    if blob_count == 0 or centroid_x is None:
        return "SCAN — no clear boundary detected", "no reliable orange boundary response"
    if centroid_x < 0.42:
        return "STEER RIGHT GENTLY", "boundary mass is left of the camera center"
    if centroid_x > 0.58:
        return "STEER LEFT GENTLY", "boundary mass is right of the camera center"
    return "KEEP CENTERED", "boundary mass is centered in the camera view"


def analyze_video(input_path: str | Path, sample_hz: float = 5.0) -> dict:
    input_path = Path(input_path)
    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {input_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 25.0)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = frame_count / fps if frame_count else 0.0
    stride = max(1, round(fps / sample_hz))
    signals: list[dict] = []
    frame_index = 0

    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if frame_index % stride != 0:
            frame_index += 1
            continue
        # Downsample for fast, deterministic browser-demo preprocessing.
        height, width = frame.shape[:2]
        scale = min(1.0, 720.0 / max(width, 1))
        if scale < 1.0:
            frame = cv2.resize(frame, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)
        height, width = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # Orange traffic cones: broad hue range to survive the bright outdoor clip.
        mask = cv2.inRange(hsv, np.array([0, 95, 90], dtype=np.uint8), np.array([28, 255, 255], dtype=np.uint8))
        # This source is an in-cabin view. Keep only the forward windshield
        # region; otherwise the airbag label and the driver's red jacket are
        # valid orange pixels but not road boundaries.
        windshield_roi = np.zeros((height, width), dtype=np.uint8)
        roi_polygon = np.array([
            [0, round(height * 0.25)],
            [round(width * 0.66), round(height * 0.25)],
            [round(width * 0.66), round(height * 0.80)],
            [0, round(height * 0.80)],
        ], dtype=np.int32)
        cv2.fillPoly(windshield_roi, [roi_polygon], 255)
        mask = cv2.bitwise_and(mask, windshield_roi)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_area = max(30.0, width * height * 0.00035)
        blobs = []
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < min_area:
                continue
            moments = cv2.moments(contour)
            if moments["m00"] <= 0:
                continue
            cx = moments["m10"] / moments["m00"] / width
            cy = moments["m01"] / moments["m00"] / height
            x, y, w, h = cv2.boundingRect(contour)
            blobs.append({"x": round(x / width, 4), "y": round(y / height, 4), "w": round(w / width, 4), "h": round(h / height, 4), "area": round(area / (width * height), 6), "cx": round(cx, 4), "cy": round(cy, 4)})
        area_ratio = float(cv2.countNonZero(mask)) / float(width * height)
        centroid_x = None
        if blobs:
            weights = np.array([b["area"] for b in blobs], dtype=float)
            centroid_x = float(np.average([b["cx"] for b in blobs], weights=weights))
        instruction, reason = _instruction(centroid_x, area_ratio, len(blobs))
        signals.append({
            "t": round(frame_index / fps, 3),
            "frame": frame_index,
            "orange_area_ratio": round(area_ratio, 6),
            "boundary_centroid_x": None if centroid_x is None else round(centroid_x, 4),
            "boundary_count": len(blobs),
            "boundaries": blobs,
            "instruction": instruction,
            "reason": reason,
            "confidence": round(min(0.98, 0.35 + min(area_ratio * 25.0, 0.5) + min(len(blobs), 3) * 0.05), 3),
        })
        frame_index += 1
    capture.release()

    return {
        "project": "ihateparking",
        "mode": "video_assist_v1",
        "language": "English",
        "source": {
            "file": "assets/pexels_parking_pov.mp4",
            "provider": "Pexels",
            "asset_id": "18495959",
            "duration_s": round(duration, 3),
            "fps": fps,
            "analysis": "orange boundary/cone perception from a fixed in-cabin view",
        },
        "limitations": [
            "This clip contains a cone driving course, not a complete parking manoeuvre.",
            "The camera does not expose a reliable reverse view or real steering-angle/CAN signal.",
            "Instructions are human-in-the-loop suggestions, not vehicle actuation commands.",
        ],
        "signals": signals,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze a video for ihateparking human-in-the-loop guidance")
    parser.add_argument("input", help="input video path")
    parser.add_argument("--output", default="artifacts/footage_assist.json")
    parser.add_argument("--sample-hz", type=float, default=5.0)
    args = parser.parse_args()
    result = analyze_video(args.input, args.sample_hz)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"[video-assist] analyzed={len(result['signals'])} samples output={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
