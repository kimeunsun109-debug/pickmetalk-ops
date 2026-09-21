"""MediaPipe Face Landmarker wrapper."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# Face oval contour (478-landmark topology).
FACE_OVAL = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
]

# Stable alignment anchors (eyes, nose, mouth).
ALIGN_INDICES = [
    33, 263,   # outer eye corners
    133, 362,  # inner eye corners
    1,         # nose tip
    61, 291,   # mouth corners
    199,       # chin
    10, 152,   # forehead / chin vertical
]


@dataclass
class FaceLandmarks:
    points: np.ndarray  # (N, 2) float32 pixel coords
    image_size: tuple[int, int]  # (width, height)


class FaceLandmarkerService:
    def __init__(self, model_path: Path) -> None:
        options = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=1,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)

    def detect(self, image_bgr: np.ndarray) -> FaceLandmarks | None:
        h, w = image_bgr.shape[:2]
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect(mp_image)
        if not result.face_landmarks:
            return None
        lm = result.face_landmarks[0]
        pts = np.array([(p.x * w, p.y * h) for p in lm], dtype=np.float32)
        return FaceLandmarks(points=pts, image_size=(w, h))

    def close(self) -> None:
        self._landmarker.close()


def oval_points(landmarks: FaceLandmarks) -> np.ndarray:
    return landmarks.points[FACE_OVAL]


def align_points(landmarks: FaceLandmarks) -> np.ndarray:
    return landmarks.points[ALIGN_INDICES]


LEFT_EYE_IDX = (33, 133, 159, 145)
RIGHT_EYE_IDX = (263, 362, 386, 374)
NOSE_TIP_IDX = 1


def _eye_center(pts: np.ndarray, indices: tuple[int, ...]) -> np.ndarray:
    return pts[list(indices)].mean(axis=0)


def pose_metrics(landmarks: FaceLandmarks) -> tuple[float, float]:
    """Return (eye_tilt_deg, nose_horizontal_offset_ratio)."""
    pts = landmarks.points
    le = _eye_center(pts, LEFT_EYE_IDX)
    re = _eye_center(pts, RIGHT_EYE_IDX)
    nose = pts[NOSE_TIP_IDX]
    eye_dist = float(np.linalg.norm(re - le)) + 1e-6
    tilt = float(np.degrees(np.arctan2(re[1] - le[1], re[0] - le[0])))
    mid = (le + re) / 2
    nose_off = float(abs(nose[0] - mid[0]) / eye_dist)
    return tilt, nose_off


def is_frontal(
    landmarks: FaceLandmarks,
    *,
    max_eye_tilt_deg: float = 10.0,
    max_nose_offset: float = 0.2,
) -> bool:
    tilt, nose_off = pose_metrics(landmarks)
    return abs(tilt) <= max_eye_tilt_deg and nose_off <= max_nose_offset
