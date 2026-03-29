"""
Face recognition: YuNet (FaceDetectorYN) inside person crops, SFace (FaceRecognizerSF)
for 128-D embeddings, cosine match against ``FaceDatabase``.

OpenCV Zoo ONNX files on GitHub are stored with Git LFS; reliable HTTP downloads use the
official Hugging Face mirrors under the ``opencv`` org (same model blobs).
"""

from __future__ import annotations

import threading
import urllib.request
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

import config

if TYPE_CHECKING:
    from database import FaceDatabase

# --- Model files (same blobs as opencv/opencv_zoo on GitHub; GitHub raw is often Git LFS) ---
YUNET_FILENAME = "face_detection_yunet_2023mar.onnx"
SFACE_FILENAME = "face_recognition_sface_2021dec.onnx"

# Hugging Face mirrors the OpenCV Zoo ONNX weights with plain HTTP (reliable).
# Canonical GitHub paths (for git clone / manual copy):
#   models/face_detection_yunet/face_detection_yunet_2023mar.onnx
#   models/face_recognition_sface/face_recognition_sface_2021dec.onnx
YUNET_DOWNLOAD_URLS = (
    f"https://huggingface.co/opencv/face_detection_yunet/resolve/main/{YUNET_FILENAME}",
)

SFACE_DOWNLOAD_URLS = (
    f"https://huggingface.co/opencv/face_recognition_sface/resolve/main/{SFACE_FILENAME}",
)

Frame = np.ndarray
PersonBox = List[float]  # [x1, y1, x2, y2] xyxy
RecognitionRecord = Dict[str, Any]


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(x))
    if n < 1e-12:
        return x.astype(np.float32)
    return (x / n).astype(np.float32)


def _is_lfs_pointer(data: bytes) -> bool:
    return data.startswith(b"version https://git-lfs.github.com/")


def _onnx_file_usable(path: Path, min_bytes: int = 2000) -> bool:
    """Reject missing files, Git LFS pointer stubs, and truncated downloads."""
    if not path.is_file():
        return False
    try:
        sz = path.stat().st_size
    except OSError:
        return False
    if sz < min_bytes:
        return False
    with open(path, "rb") as f:
        head = f.read(80)
    if _is_lfs_pointer(head):
        return False
    return True


def _download_to_path(url: str, dest: Path, timeout_s: int = 300) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (compatible; CVPipeline/1.0)"},
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        data = resp.read()
    if len(data) < 500 or _is_lfs_pointer(data):
        raise RuntimeError(
            f"Download from {url} looks invalid (LFS pointer or too small). "
            "Try another mirror or place the ONNX file manually."
        )
    dest.write_bytes(data)


def ensure_face_models(
    yunet_path: Optional[Path] = None,
    sface_path: Optional[Path] = None,
) -> Tuple[Path, Path]:
    """
    Ensure YuNet and SFace ONNX files exist under ``models/weights/``.
    Tries Hugging Face first, then GitHub raw (may fail for LFS).
    """
    yp = Path(yunet_path) if yunet_path is not None else config.YUNET_ONNX_PATH
    sp = Path(sface_path) if sface_path is not None else config.SFACE_ONNX_PATH

    if not _onnx_file_usable(yp):
        if yp.is_file():
            try:
                yp.unlink()
            except OSError:
                pass
        last_err: Optional[BaseException] = None
        for url in YUNET_DOWNLOAD_URLS:
            try:
                _download_to_path(url, yp)
                break
            except BaseException as e:
                last_err = e
        else:
            raise RuntimeError(f"Could not download YuNet ONNX to {yp}") from last_err

    if not _onnx_file_usable(sp, min_bytes=10_000):
        if sp.is_file():
            try:
                sp.unlink()
            except OSError:
                pass
        last_err = None
        for url in SFACE_DOWNLOAD_URLS:
            try:
                _download_to_path(url, sp)
                break
            except BaseException as e:
                last_err = e
        else:
            raise RuntimeError(f"Could not download SFace ONNX to {sp}") from last_err

    return yp, sp


def _clamp_xyxy(
    x1: float, y1: float, x2: float, y2: float, w: int, h: int
) -> Tuple[int, int, int, int]:
    x1i = int(max(0, min(round(x1), w - 1)))
    y1i = int(max(0, min(round(y1), h - 1)))
    x2i = int(max(0, min(round(x2), w)))
    y2i = int(max(0, min(round(y2), h)))
    if x2i <= x1i:
        x2i = min(w, x1i + 1)
    if y2i <= y1i:
        y2i = min(h, y1i + 1)
    return x1i, y1i, x2i, y2i


def _largest_face_row(faces: np.ndarray) -> Optional[np.ndarray]:
    if faces is None or len(faces) == 0:
        return None
    best_idx = 0
    best_area = -1.0
    for i in range(len(faces)):
        fw, fh = float(faces[i][2]), float(faces[i][3])
        area = fw * fh
        if area > best_area:
            best_area = area
            best_idx = i
    return faces[best_idx]


class FaceEmbeddingPipeline:
    """
    YuNet + FaceRecognizerSF: alignCrop + feature, L2-normalized embedding for cosine match.
    Thread-safe for concurrent calls (single lock around OpenCV inference).
    """

    def __init__(
        self,
        yunet_path: Optional[Path] = None,
        sface_path: Optional[Path] = None,
    ) -> None:
        if not hasattr(cv2, "FaceRecognizerSF"):
            raise RuntimeError(
                "cv2.FaceRecognizerSF is missing. Install opencv-contrib-python "
                "(standard opencv-python does not include FaceRecognizerSF)."
            )

        yp, sp = ensure_face_models(yunet_path, sface_path)
        self._yunet_path = yp
        self._sface_path = sp

        h, w = config.YUNET_INPUT_SIZE[1], config.YUNET_INPUT_SIZE[0]
        self._detector = cv2.FaceDetectorYN.create(
            str(yp),
            "",
            (w, h),
            config.YUNET_SCORE_THRESHOLD,
            config.YUNET_NMS_THRESHOLD,
            config.YUNET_TOP_K,
        )
        self._recognizer = cv2.FaceRecognizerSF.create(str(sp), "")
        self._lock = threading.Lock()

    def embed_from_bgr(self, bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect the largest face, align, extract SFace feature.

        Returns
        -------
        embedding : np.ndarray
            L2-normalized feature vector (typically 128-D).
        face_row : np.ndarray
            YuNet row (15,) for this face (crop coordinates).
        """
        if bgr is None or bgr.size == 0:
            raise RuntimeError("Empty image")

        hc, wc = bgr.shape[:2]
        with self._lock:
            self._detector.setInputSize((wc, hc))
            _, faces = self._detector.detect(bgr)

            if faces is None or len(faces) == 0:
                raise RuntimeError("No face detected in image")

            face_row = _largest_face_row(faces)
            if face_row is None:
                raise RuntimeError("No face detected in image")

            fr = np.asarray(face_row, dtype=np.float32).reshape(1, -1)
            aligned = self._recognizer.alignCrop(bgr, fr)
            feat = self._recognizer.feature(aligned)

        vec = np.asarray(feat).reshape(-1).astype(np.float32)
        return _l2_normalize(vec), face_row

    def embed_from_bgr_optional(
        self, bgr: np.ndarray
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        try:
            return self.embed_from_bgr(bgr)
        except RuntimeError:
            return None


_pipeline_singleton: Optional["FaceEmbeddingPipeline"] = None
_pipeline_lock = threading.Lock()


def get_face_embedding_pipeline() -> FaceEmbeddingPipeline:
    """Process-wide shared pipeline (used by ``FaceDatabase`` and ``FaceRecognizer``)."""
    global _pipeline_singleton
    with _pipeline_lock:
        if _pipeline_singleton is None:
            _pipeline_singleton = FaceEmbeddingPipeline()
        return _pipeline_singleton


class FaceRecognizer:
    """
    Person crops -> YuNet -> SFace embedding -> cosine match vs ``FaceDatabase``.
    """

    def __init__(
        self,
        pipeline: Optional[FaceEmbeddingPipeline] = None,
    ) -> None:
        self._pipeline = pipeline or get_face_embedding_pipeline()

    def recognize(
        self,
        frame: Frame,
        person_boxes: Sequence[Any],
        database: "FaceDatabase",
    ) -> List[RecognitionRecord]:
        """
        For each person box, crop the frame, detect a face, embed, match with cosine similarity.

        Parameters
        ----------
        frame
            BGR image, shape (H, W, 3).
        person_boxes
            Each entry is ``[x1, y1, x2, y2]`` or ``([x1, y1, x2, y2], person_conf)``.
        database
            Loaded ``FaceDatabase`` with embeddings from the same SFace pipeline.

        Returns
        -------
        list of dicts with keys:
            ``person_box`` ([x1,y1,x2,y2] floats),
            ``person_score`` (optional detector confidence if provided),
            ``label`` (str),
            ``face_box`` ([x1,y1,x2,y2] in full frame, or None if no face),
            ``match_distance`` (cosine distance of best gallery hit, or inf if N/A),
        """
        if frame is None or frame.size == 0:
            return []

        fh, fw = frame.shape[:2]
        normalized = _normalize_person_inputs(person_boxes)
        out: List[RecognitionRecord] = []

        for pbox, pscore, pclass_id, pclass_name in normalized:
            base_label = pclass_name if pclass_name else "Unknown"
            x1, y1, x2, y2 = _clamp_xyxy(*pbox, fw, fh)
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                out.append(
                    {
                        "person_box": [float(pbox[0]), float(pbox[1]), float(pbox[2]), float(pbox[3])],
                        "person_score": pscore,
                        "class_id": pclass_id,
                        "class_name": pclass_name,
                        "identity": "Unknown",
                        "recognized": False,
                        "label": "Unknown - Customer",
                        "face_box": None,
                        "match_distance": float("inf"),
                    }
                )
                continue

            emb_face = self._pipeline.embed_from_bgr_optional(crop)
            if emb_face is None:
                out.append(
                    {
                        "person_box": [float(pbox[0]), float(pbox[1]), float(pbox[2]), float(pbox[3])],
                        "person_score": pscore,
                        "class_id": pclass_id,
                        "class_name": pclass_name,
                        "identity": "Unknown",
                        "recognized": False,
                        "label": "Unknown - Customer",
                        "face_box": None,
                        "match_distance": float("inf"),
                    }
                )
                continue

            embedding, face_row = emb_face
            # YuNet box is x,y,w,h in *crop* coordinates
            fx, fy, fw_i, fh_i = float(face_row[0]), float(face_row[1]), float(face_row[2]), float(face_row[3])
            gx1 = fx + x1
            gy1 = fy + y1
            gx2 = gx1 + fw_i
            gy2 = gy1 + fh_i
            face_xyxy = _clamp_xyxy(gx1, gy1, gx2, gy2, fw, fh)
            face_box_frame = [float(face_xyxy[0]), float(face_xyxy[1]), float(face_xyxy[2]), float(face_xyxy[3])]

            best_name, dist = database.match(embedding, metric="cosine")
            if (
                best_name is not None
                and dist <= config.COSINE_MATCH_THRESHOLD
            ):
                identity = best_name
                recognized = True
            else:
                identity = "Unknown"
                recognized = False

            role = "Staff" if recognized else "Customer"
            label = f"{identity} - {role}"

            out.append(
                {
                    "person_box": [float(pbox[0]), float(pbox[1]), float(pbox[2]), float(pbox[3])],
                    "person_score": pscore,
                    "class_id": pclass_id,
                    "class_name": pclass_name,
                    "identity": identity,
                    "recognized": recognized,
                    "label": label,
                    "face_box": face_box_frame,
                    "match_distance": float(dist),
                }
            )

        return out


def _normalize_person_inputs(
    person_boxes: Sequence[Any],
) -> List[Tuple[PersonBox, Optional[float], Optional[int], Optional[str]]]:
    out: List[Tuple[PersonBox, Optional[float], Optional[int], Optional[str]]] = []

    def _coerce_box(raw: Any) -> PersonBox:
        a = np.asarray(raw, dtype=np.float64).ravel()
        if a.size != 4:
            raise ValueError("Each person box must have 4 values [x1, y1, x2, y2]")
        return [float(a[0]), float(a[1]), float(a[2]), float(a[3])]

    for item in person_boxes:
        score: Optional[float] = None
        class_id: Optional[int] = None
        class_name: Optional[str] = None
        if isinstance(item, dict):
            box_src = item.get("box", item.get("person_box"))
            if box_src is None:
                raise ValueError("Detection dict must include 'box' or 'person_box'")
            if item.get("score") is not None:
                score = float(item["score"])
            if item.get("class_id") is not None:
                class_id = int(item["class_id"])
            if item.get("class_name") is not None:
                class_name = str(item["class_name"])
            out.append((_coerce_box(box_src), score, class_id, class_name))
            continue

        if isinstance(item, (list, tuple)) and len(item) == 2:
            try:
                score = float(item[1])
                out.append((_coerce_box(item[0]), score, class_id, class_name))
                continue
            except (TypeError, ValueError):
                pass
        out.append((_coerce_box(item), None, class_id, class_name))
    return out
