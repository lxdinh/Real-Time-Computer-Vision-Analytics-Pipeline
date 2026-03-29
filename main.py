"""
Real-Time Computer Vision Analytics Pipeline — webcam entry point.

Person detection (YOLOv11) → face embedding + gallery match (YuNet + SFace).
"""

from __future__ import annotations

import math
import time

import cv2

import config
from database import FaceDatabase
from detector import StaffDetector
from recognizer import FaceRecognizer, ensure_face_models


def _draw_readable_text(
    frame,
    text: str,
    x: int,
    y: int,
    *,
    font_scale: float = 0.6,
    fg=(255, 255, 255),
    bg=(0, 0, 0),
    thickness: int = 2,
) -> None:
    """Draw text with a solid background box for high-contrast readability."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x0 = max(0, x - 3)
    y0 = max(0, y - th - baseline - 5)
    x1 = min(frame.shape[1] - 1, x + tw + 3)
    y1 = min(frame.shape[0] - 1, y + baseline + 2)
    cv2.rectangle(frame, (x0, y0), (x1, y1), bg, -1)
    cv2.putText(frame, text, (x, y), font, font_scale, fg, thickness, cv2.LINE_AA)


def main() -> None:
    ensure_face_models()

    database = FaceDatabase()
    database.load_from_directory(config.KNOWN_FACES_DIR)

    detector = StaffDetector()
    recognizer = FaceRecognizer()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam index 0")

    window = "Real-Time Computer Vision Analytics Pipeline"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    # BGR
    color_person = (0, 255, 0)
    color_face = (255, 0, 0)
    color_text = (0, 255, 255)

    t_prev = time.perf_counter()
    fps_smooth = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break

            t_now = time.perf_counter()
            dt = t_now - t_prev
            t_prev = t_now
            if dt > 1e-6:
                inst_fps = 1.0 / dt
                fps_smooth = 0.9 * fps_smooth + 0.1 * inst_fps if fps_smooth > 0 else inst_fps

            detections = detector.detect(frame)
            yolo_hit = len(detections) > 0
            used_fallback = False
            if (
                not detections
                and config.DEBUG_FALLBACK_FULL_FRAME_WHEN_NO_DETECTIONS
            ):
                h, w = frame.shape[:2]
                used_fallback = True
                detections = [
                    {
                        "box": [0.0, 0.0, float(w - 1), float(h - 1)],
                        "score": 1.0,
                        "class_id": -1,
                        "class_name": "Person",
                    }
                ]
            results = recognizer.recognize(frame, detections, database)

            for rec in results:
                px1, py1, px2, py2 = [int(round(x)) for x in rec["person_box"]]
                cv2.rectangle(frame, (px1, py1), (px2, py2), color_person, 2)

                fb = rec.get("face_box")
                if fb is not None:
                    fx1, fy1, fx2, fy2 = [int(round(x)) for x in fb]
                    cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), color_face, 2)

                # Final text is strictly Face-ID-driven (YOLO class is ignored for role):
                # recognized => Staff, unknown => Customer.
                identity = str(rec.get("identity", "Unknown"))
                recognized = bool(rec.get("recognized", False))
                role = "Staff" if recognized else "Customer"
                label = f"{identity} - {role}"
                # Requested color mapping: Staff=red, Customer=green (BGR).
                label_fg = (0, 0, 255) if role == "Staff" else (0, 255, 0)
                label_y = py1 - 10 if py1 >= 28 else py1 + 22
                _draw_readable_text(
                    frame,
                    label,
                    px1,
                    label_y,
                    font_scale=0.65,
                    fg=label_fg,
                    bg=(0, 0, 0),
                    thickness=2,
                )

                # Debug observability: match distance on-screen + terminal logging.
                d = float(rec.get("match_distance", float("inf")))
                has_face = rec.get("face_box") is not None
                dist_text = f"d={d:.2f}" if math.isfinite(d) else "d=NA"

                if config.DEBUG_SHOW_MATCH_DISTANCE:
                    dist_y = min(frame.shape[0] - 8, py1 + 18)
                    _draw_readable_text(
                        frame,
                        dist_text,
                        px1,
                        dist_y,
                        font_scale=0.55,
                        fg=(0, 255, 255),
                        bg=(0, 0, 0),
                        thickness=2,
                    )

                if config.DEBUG_PRINT_MATCH_DISTANCE and has_face:
                    print(
                        f"[face-match] label={label} {dist_text}",
                        flush=True,
                    )

            if config.DEBUG_SHOW_MATCH_DISTANCE:
                if yolo_hit:
                    status_text = f"YOLO: HIT ({len(detections)})"
                    status_color = (0, 255, 0)
                elif used_fallback:
                    status_text = "YOLO: MISS (fallback=ON)"
                    status_color = (0, 165, 255)
                else:
                    status_text = "YOLO: MISS"
                    status_color = (0, 0, 255)

                cv2.putText(
                    frame,
                    status_text,
                    (10, 56),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    status_color,
                    2,
                    cv2.LINE_AA,
                )

            cv2.putText(
                frame,
                f"FPS: {fps_smooth:.1f}",
                (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color_text,
                2,
                cv2.LINE_AA,
            )

            cv2.imshow(window, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
