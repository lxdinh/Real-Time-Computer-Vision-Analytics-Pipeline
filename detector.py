"""
Staff/Customer detection using Ultralytics YOLOv11.

Returns per-detection metadata with custom class mapping so downstream code can
display correct labels and run face recognition only for staff class.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

import config

# OpenCV BGR ndarray or path accepted by ultralytics internally; we only pass ndarray.
Frame = np.ndarray
BoxXYXY = List[float]  # [x1, y1, x2, y2]
Detection = Dict[str, Any]


class StaffDetector:
    """
    Wraps YOLOv11 for custom-class detection (Customer/Staff).

    Parameters
    ----------
    weights_path
        Path to model weights. Defaults to ``config.YOLO_WEIGHTS_PATH``.
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
        self._class_names = getattr(
            config,
            "YOLO_CLASS_NAMES",
            {0: "Customer", 1: "Staff"},
        )
        self._allowed_class_ids = sorted(int(k) for k in self._class_names.keys())
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
        Run inference on one BGR frame and return custom-class detections.

        Returns
        -------
        list of dict
            Each item has:
            - box: [x1, y1, x2, y2]
            - score: confidence
            - class_id: integer class id
            - class_name: display label
        """
        if frame is None or frame.size == 0:
            return []

        # Restrict decode to relevant classes for lower overhead.
        kwargs = {
            "conf": self._conf,
            "iou": self._iou,
            "classes": self._allowed_class_ids,
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
                class_id = int(clss[i])
                if class_id not in self._class_names:
                    continue
                box = [float(xyxy[i, j]) for j in range(4)]
                score = float(confs[i])
                out.append(
                    {
                        "box": box,
                        "score": score,
                        "class_id": class_id,
                        "class_name": str(self._class_names[class_id]),
                    }
                )
        return out
