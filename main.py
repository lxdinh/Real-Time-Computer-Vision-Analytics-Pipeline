"""
Real-Time Computer Vision Analytics Pipeline — webcam entry point.

Person detection (YOLOv11) → face embedding + gallery match (YuNet + SFace).
"""

from __future__ import annotations

import time

import cv2

import config
from database import FaceDatabase
from detector import StaffDetector
from recognizer import FaceRecognizer, ensure_face_models


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
            results = recognizer.recognize(frame, detections, database)

            for rec in results:
                px1, py1, px2, py2 = [int(round(x)) for x in rec["person_box"]]
                cv2.rectangle(frame, (px1, py1), (px2, py2), color_person, 2)

                fb = rec.get("face_box")
                if fb is not None:
                    fx1, fy1, fx2, fy2 = [int(round(x)) for x in fb]
                    cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), color_face, 2)

                label = rec["label"]
                cv2.putText(
                    frame,
                    label,
                    (px1, max(0, py1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color_text,
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
