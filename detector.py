"""
Person / staff detection using Ultralytics YOLOv11.

Filters to COCO class ``person`` (ID 0) only. Tuned for low-latency use inside a
tight video loop (single-frame inference, no verbose logging).
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

import config

# OpenCV BGR ndarray or path accepted by ultralytics internally; we only pass ndarray.
Frame = np.ndarray
BoxXYXY = List[float]  # [x1, y1, x2, y2]
Detection = Tuple[BoxXYXY, float]


class StaffDetector:
    """
    Wraps YOLOv11 for single-class (person) detection.

    Parameters
    ----------
    weights_path
        Path to ``yolo11n.pt``. Defaults to ``config.YOLO_WEIGHTS_PATH``.
    conf
        Minimum confidence. Defaults to ``config.YOLO_CONFIDENCE_THRESHOLD`` or 0.5.
    iou
        NMS IoU threshold. Defaults to ``config.YOLO_IOU_THRESHOLD``.
    device
        ``"cpu"``, ``"0"``, ``"cuda:0"``, etc. ``None`` lets Ultralytics auto-select.
    """

    def __init__(
        self,
        weights_path: Optional[Union[str, Path]] = None,
        conf: Optional[float] = None,
        iou: Optional[float] = None,
        device: Optional[str] = None,
    ) -> None:
        from ultralytics import YOLO  # lazy import keeps CLI tools import-light

        path = Path(weights_path) if weights_path is not None else config.YOLO_WEIGHTS_PATH
        self._weights_path = path

        self._conf = (
            conf
            if conf is not None
            else getattr(config, "YOLO_CONFIDENCE_THRESHOLD", 0.5)
        )
        self._iou = (
            iou if iou is not None else getattr(config, "YOLO_IOU_THRESHOLD", 0.45)
        )
        self._person_class = getattr(config, "YOLO_PERSON_CLASS_ID", 0)
        self._device = device

        self._model = YOLO(str(path))
        if self._device is not None:
            self._model.to(self._device)

    @property
    def confidence_threshold(self) -> float:
        return self._conf

    @confidence_threshold.setter
    def confidence_threshold(self, value: float) -> None:
        self._conf = float(value)

    def detect(self, frame: Frame) -> List[Detection]:
        """
        Run inference on one BGR frame and return person boxes only.

        Returns
        -------
        list of ( [x1, y1, x2, y2], confidence )
            Pixel coordinates in the same space as ``frame`` (xyxy, inclusive-style).
        """
        if frame is None or frame.size == 0:
            return []

        # classes=[0] => only "person" in COCO; skips decoding other heads earlier
        kwargs = {
            "conf": self._conf,
            "iou": self._iou,
            "classes": [self._person_class],
            "verbose": False,
            "stream": False,
        }
        if self._device is not None:
            kwargs["device"] = self._device

        results = self._model.predict(frame, **kwargs)
        return self._parse_results(results)

    def _parse_results(self, results: Sequence[object]) -> List[Detection]:
        out: List[Detection] = []
        for r in results:
            boxes = getattr(r, "boxes", None)
            if boxes is None or len(boxes) == 0:
                continue
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            clss = boxes.cls.cpu().numpy().astype(int)

            for i in range(len(boxes)):
                if clss[i] != self._person_class:
                    continue
                box = [float(xyxy[i, j]) for j in range(4)]
                score = float(confs[i])
                out.append((box, score))
        return out
