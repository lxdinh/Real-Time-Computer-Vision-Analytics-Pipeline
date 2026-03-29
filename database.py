"""
Lightweight face gallery: load registered identities from disk, extract SFace
embeddings (FaceRecognizerSF, same pipeline as ``recognizer.py``), and match by
cosine similarity or L2 distance.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

import config

logger = logging.getLogger(__name__)


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x)
    if n < 1e-12:
        return x
    return (x / n).astype(np.float32)


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """1 - cosine_similarity; 0 = identical direction."""
    a = _l2_normalize(a.flatten())
    b = _l2_normalize(b.flatten())
    return float(1.0 - np.dot(a, b))


def _l2_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a.flatten() - b.flatten()))


class FaceDatabase:
    """
    In-memory store: identity -> L2-normalized embedding (float32 vector).
    Images are expected under ``known_faces/<identity_name>/*`` when
    ``config.REQUIRE_SUBFOLDERS_FOR_IDENTITIES`` is True.
    """

    def __init__(self) -> None:
        self._identities: List[str] = []
        self._embeddings: Dict[str, np.ndarray] = {}

    def embedding_from_bgr(self, bgr: np.ndarray) -> np.ndarray:
        """
        Detect the largest face, align with SFace, return L2-normalized embedding.
        """
        from recognizer import get_face_embedding_pipeline

        emb, _ = get_face_embedding_pipeline().embed_from_bgr(bgr)
        return emb

    def load_from_directory(
        self,
        root: Optional[Path] = None,
        *,
        extensions: Optional[set] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Walk ``root`` (default: config.KNOWN_FACES_DIR), load images per identity,
        compute mean embedding per identity, and store in memory.

        Returns a copy of the identity -> embedding map.
        """
        base = Path(root) if root is not None else config.KNOWN_FACES_DIR
        ext = extensions or config.GALLERY_IMAGE_EXTENSIONS

        if not base.is_dir():
            raise FileNotFoundError(f"Known faces directory does not exist: {base}")

        self._identities.clear()
        self._embeddings.clear()

        if config.REQUIRE_SUBFOLDERS_FOR_IDENTITIES:
            for person_dir in sorted(p for p in base.iterdir() if p.is_dir()):
                label = person_dir.name
                imgs: List[Path] = []
                for p in person_dir.rglob("*"):
                    if p.is_file() and p.suffix.lower() in ext:
                        imgs.append(p)
                if not imgs:
                    logger.warning("Skipping %s: no images", person_dir)
                    continue
                embs: List[np.ndarray] = []
                for img_path in imgs:
                    bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
                    if bgr is None:
                        logger.warning("Unreadable image: %s", img_path)
                        continue
                    try:
                        embs.append(self.embedding_from_bgr(bgr))
                    except RuntimeError as e:
                        logger.warning("%s: %s", img_path, e)
                if not embs:
                    logger.warning("No valid embeddings for identity %s", label)
                    continue
                mean_emb = _l2_normalize(np.mean(np.stack(embs, axis=0), axis=0))
                self._identities.append(label)
                self._embeddings[label] = mean_emb
        else:
            for p in sorted(base.iterdir()):
                if not p.is_file() or p.suffix.lower() not in ext:
                    continue
                label = p.stem
                bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
                if bgr is None:
                    logger.warning("Unreadable image: %s", p)
                    continue
                try:
                    emb = self.embedding_from_bgr(bgr)
                except RuntimeError as e:
                    logger.warning("%s: %s", p, e)
                    continue
                self._identities.append(label)
                self._embeddings[label] = emb

        return dict(self._embeddings)

    @property
    def identities(self) -> List[str]:
        return list(self._identities)

    def get_embedding(self, identity: str) -> np.ndarray:
        if identity not in self._embeddings:
            raise KeyError(identity)
        return self._embeddings[identity].copy()

    def match(
        self,
        query_embedding: np.ndarray,
        *,
        metric: Optional[str] = None,
    ) -> Tuple[Optional[str], float]:
        """
        Return best-matching identity and distance (lower is better for both metrics).
        Distance is cosine distance or L2 depending on ``metric`` / config.
        """
        if not self._embeddings:
            return None, float("inf")

        m = metric or config.SIMILARITY_METRIC
        q = query_embedding.astype(np.float32).flatten()

        best_name: Optional[str] = None
        best_dist = float("inf")

        for name, ref in self._embeddings.items():
            if m == "cosine":
                d = _cosine_distance(q, ref)
            elif m == "l2":
                d = _l2_distance(q, ref)
            else:
                raise ValueError(f"Unknown metric: {m}")
            if d < best_dist:
                best_dist = d
                best_name = name

        return best_name, best_dist

    def is_match(self, query_embedding: np.ndarray, metric: Optional[str] = None) -> bool:
        _, dist = self.match(query_embedding, metric=metric)
        m = metric or config.SIMILARITY_METRIC
        if m == "cosine":
            return dist <= config.COSINE_MATCH_THRESHOLD
        if m == "l2":
            return dist <= config.L2_MATCH_THRESHOLD
        raise ValueError(f"Unknown metric: {m}")
