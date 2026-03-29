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

# --- Staff/Customer detection (Ultralytics YOLOv11 custom classes) ---
# Your trained weights live at models/best.pt
YOLO_WEIGHTS_PATH = MODELS_DIR / "best.pt"
# Custom class mapping from your dataset
YOLO_CLASS_NAMES = {
    0: "Customer",
    1: "Staff",
}
YOLO_STAFF_CLASS_ID = 1
YOLO_CONFIDENCE_THRESHOLD = 0.3
YOLO_IOU_THRESHOLD = 0.45  # built-in NMS; tune for crowded scenes

# TensorRT ``.engine`` files from ``export_trt.py`` (build on target Jetson hardware)
TRT_ENGINES_DIR = MODELS_DIR / "engines"

# --- Face detection (YuNet via FaceDetectorYN) ---
YUNET_INPUT_SIZE = (640, 640)  # max side / internal resize hint; setInputSize uses frame dims
YUNET_SCORE_THRESHOLD = 0.6
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
COSINE_MATCH_THRESHOLD = 0.5  # lower distance = more similar; tune on your data
L2_MATCH_THRESHOLD = 1.0

# --- Runtime / threading (reserved for Step 4) ---
TARGET_FRAME_HEIGHT = 720
CAPTURE_FPS_HINT = 30

# --- Debug / observability ---
# Show match distance on the video overlay (e.g., d=0.45)
DEBUG_SHOW_MATCH_DISTANCE = True
# Print match distance in terminal for every detected face
DEBUG_PRINT_MATCH_DISTANCE = True
# If YOLO misses in a frame, run face recognition on whole frame as fallback.
DEBUG_FALLBACK_FULL_FRAME_WHEN_NO_DETECTIONS = True
