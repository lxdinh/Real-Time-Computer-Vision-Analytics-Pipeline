#!/usr/bin/env python3
"""
Export Ultralytics YOLOv11 weights to a TensorRT ``.engine`` for edge deployment.

Run only on the Jetson (or other target) where the engine will execute — not on a dev PC.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Hardware warning — first output (before heavy imports / config)
# ---------------------------------------------------------------------------
print()
print("=" * 78)
print("  WARNING: TensorRT engines are hardware-specific.")
print("  Only run this script on the target device (e.g., NVIDIA Jetson),")
print("  NOT your development machine!")
print("=" * 78)
print()

import argparse
import sys
from pathlib import Path
from typing import Optional

import config


def export_yolo11_tensorrt(
    weights: Path,
    out_dir: Path,
    *,
    imgsz: int = 640,
    half: bool = True,
    int8: bool = True,
    data: Optional[str] = "coco8.yaml",
    device: str = "0",
) -> Path:
    """
    Load a YOLO ``.pt`` checkpoint and export ``format='engine'`` via Ultralytics.

    INT8 calibration requires a dataset YAML; the default ``coco8.yaml`` is tiny and
    ships with Ultralytics. Point ``data`` at your own images/YAML for production.

    Parameters
    ----------
    half, int8
        Passed through to ``YOLO.export`` to request FP16 and INT8 where supported
        by the TensorRT builder stack (see Ultralytics + TensorRT docs).
    """
    from ultralytics import YOLO

    if not weights.is_file():
        raise FileNotFoundError(f"Weights not found: {weights}")

    out_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(weights))

    kwargs = {
        "format": "engine",
        "imgsz": imgsz,
        "half": half,
        "int8": int8,
        "device": device,
        "project": str(out_dir),
        "name": "yolo11n_trt",
        "exist_ok": True,
    }
    if int8 and data:
        kwargs["data"] = data

    exported = model.export(**kwargs)
    path = Path(exported)
    print(f"\n[export_trt] TensorRT engine written to: {path.resolve()}")
    return path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Export yolo11n.pt to TensorRT .engine (run on target Jetson only).",
    )
    p.add_argument(
        "--weights",
        type=Path,
        default=config.YOLO_WEIGHTS_PATH,
        help="Path to yolo11n.pt (default: config.YOLO_WEIGHTS_PATH)",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=config.TRT_ENGINES_DIR,
        help="Directory for TensorRT output (default: models/engines/)",
    )
    p.add_argument("--imgsz", type=int, default=640, help="Export image size")
    p.add_argument(
        "--data",
        type=str,
        default="coco8.yaml",
        help="Dataset YAML for INT8 calibration (ignored if --no-int8). Default: coco8.yaml",
    )
    p.add_argument("--device", type=str, default="0", help="CUDA device id, e.g. 0")
    p.add_argument(
        "--no-int8",
        action="store_true",
        help="Disable INT8 (still uses --half FP16 when set)",
    )
    p.add_argument(
        "--no-half",
        action="store_true",
        help="Disable FP16 (FP32; rarely needed on Jetson)",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    int8 = not args.no_int8
    half = not args.no_half

    try:
        export_yolo11_tensorrt(
            args.weights,
            args.out_dir,
            imgsz=args.imgsz,
            half=half,
            int8=int8,
            data=args.data if int8 else None,
            device=args.device,
        )
    except Exception as e:
        print(f"\n[export_trt] FAILED: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
