"""
Central configuration for the Real-Time Computer Vision Analytics Pipeline.

Paths default to a `models/` layout compatible with the OpenCV Zoo
(face_detection_yunet, face_recognition_sface). Download weights before first run.
"""

from __future__ import annotations

from pathlib import Path

# --- Project root (this file lives at repo root) ---
PROJECT_ROOT = Path(__file__).resolve().parent

# --- Directory layout ---
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_WEIGHTS_DIR = MODELS_DIR / "weights"
KNOWN_FACES_DIR = PROJECT_ROOT / "data" / "known_faces"
OUTPUT_DIR = PROJECT_ROOT / "output"

# OpenCV Zoo ONNX models (YuNet + SFace). Auto-download in ``recognizer.ensure_face_models``.
# GitHub hosts these as Git LFS; use Hugging Face mirrors or copy from a local opencv_zoo clone.
YUNET_ONNX_PATH = MODELS_WEIGHTS_DIR / "face_detection_yunet_2023mar.onnx"
SFACE_ONNX_PATH = MODELS_WEIGHTS_DIR / "face_recognition_sface_2021dec.onnx"

# --- Staff detection (Ultralytics YOLOv11; COCO class 0 = person) ---
# Place `yolo11n.pt` in the project root, or pass a custom path to StaffDetector.
YOLO_WEIGHTS_PATH = PROJECT_ROOT / "yolo11n.pt"
YOLO_PERSON_CLASS_ID = 0
YOLO_CONFIDENCE_THRESHOLD = 0.5
YOLO_IOU_THRESHOLD = 0.45  # built-in NMS; tune for crowded scenes

# TensorRT ``.engine`` files from ``export_trt.py`` (build on target Jetson hardware)
TRT_ENGINES_DIR = MODELS_DIR / "engines"

# --- Face detection (YuNet via FaceDetectorYN) ---
YUNET_INPUT_SIZE = (640, 640)  # max side / internal resize hint; setInputSize uses frame dims
YUNET_SCORE_THRESHOLD = 0.75
YUNET_NMS_THRESHOLD = 0.45
YUNET_TOP_K = 5000

# --- Face recognition (SFace via FaceRecognizerSF; 128-D embedding) ---
SFACE_EMBEDDING_DIM = 128

# --- Database / gallery ---
# Supported image extensions when scanning KNOWN_FACES_DIR
GALLERY_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
# Subfolder name = identity label; multiple images per person are averaged
REQUIRE_SUBFOLDERS_FOR_IDENTITIES = True

# --- Matching (used when comparing embeddings; recognizer will reuse) ---
SIMILARITY_METRIC = "cosine"  # "cosine" | "l2"
COSINE_MATCH_THRESHOLD = 0.35  # lower distance = more similar; tune on your data
L2_MATCH_THRESHOLD = 1.0

# --- Runtime / threading (reserved for Step 4) ---
TARGET_FRAME_HEIGHT = 720
CAPTURE_FPS_HINT = 30
