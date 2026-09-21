"""Landmark-aligned face QA — fair identity similarity vs LOCK."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from landmarks import FaceLandmarkerService, oval_points

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
PASS_THRESHOLD = 0.85
CROP_SIZE = 32


def _embedding_from_crop(crop_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (CROP_SIZE, CROP_SIZE), interpolation=cv2.INTER_AREA)
    vec = small.astype(np.float32).flatten()
    vec = (vec - vec.min()) / (vec.max() - vec.min() + 1e-6)
    norm = float(np.linalg.norm(vec)) + 1e-6
    return vec / norm


def _face_crop(image_bgr: np.ndarray, points: np.ndarray, *, pad: float = 0.18) -> np.ndarray | None:
    h, w = image_bgr.shape[:2]
    x_min, y_min = points.min(axis=0)
    x_max, y_max = points.max(axis=0)
    bw = x_max - x_min
    bh = y_max - y_min
    if bw < 8 or bh < 8:
        return None
    px, py = bw * pad, bh * pad
    x0 = int(max(0, x_min - px))
    y0 = int(max(0, y_min - py))
    x1 = int(min(w, x_max + px))
    y1 = int(min(h, y_max + py))
    if x1 - x0 < 16 or y1 - y0 < 16:
        return None
    return image_bgr[y0:y1, x0:x1].copy()


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    dot = float(np.dot(a, b))
    return float(np.clip(dot, 0.0, 1.0))


def score_vs_lock(
    lock_bgr: np.ndarray,
    lock_points: np.ndarray,
    image_bgr: np.ndarray,
    image_points: np.ndarray,
) -> float:
    lock_crop = _face_crop(lock_bgr, lock_points)
    img_crop = _face_crop(image_bgr, image_points)
    if lock_crop is None or img_crop is None:
        return 0.0
    lock_emb = _embedding_from_crop(lock_crop)
    img_emb = _embedding_from_crop(img_crop)
    return cosine_similarity(lock_emb, img_emb)


def evaluate_image(
    landmarker: FaceLandmarkerService,
    lock_bgr: np.ndarray,
    lock_points: np.ndarray,
    image_path: Path,
) -> dict:
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        return {"path": str(image_path), "ok": False, "error": "read_failed", "similarity": 0.0, "passed": False}
    lm = landmarker.detect(image_bgr)
    if lm is None:
        return {"path": str(image_path), "ok": False, "error": "no_face", "similarity": 0.0, "passed": False}
    sim = score_vs_lock(lock_bgr, lock_points, image_bgr, oval_points(lm))
    return {
        "path": str(image_path),
        "ok": True,
        "similarity": sim,
        "percent": round(sim * 1000) / 10,
        "passed": sim >= PASS_THRESHOLD,
    }


def evaluate_batch(lock_path: Path, image_paths: list[Path]) -> dict:
    lock_bgr = cv2.imread(str(lock_path), cv2.IMREAD_COLOR)
    if lock_bgr is None:
        raise FileNotFoundError(lock_path)
    landmarker = FaceLandmarkerService(MODELS_DIR / "face_landmarker.task")
    try:
        lock_lm = landmarker.detect(lock_bgr)
        if lock_lm is None:
            raise RuntimeError(f"No face in LOCK: {lock_path}")
        lock_pts = oval_points(lock_lm)
        rows = [evaluate_image(landmarker, lock_bgr, lock_pts, p) for p in image_paths]
    finally:
        landmarker.close()
    rows.sort(key=lambda r: r.get("similarity", 0), reverse=True)
    passed = sum(1 for r in rows if r.get("passed"))
    return {
        "lock": str(lock_path),
        "method": "landmark_aligned_face_crop",
        "passThreshold": PASS_THRESHOLD,
        "passed": passed,
        "total": len(rows),
        "results": rows,
    }
