"""Affine alignment of LOCK face patch onto target pose."""

from __future__ import annotations

import cv2
import numpy as np

from landmarks import FaceLandmarks, align_points

# MediaPipe landmark indices (subject-centric).
LEFT_EYE = (33, 133, 159, 145)
RIGHT_EYE = (263, 362, 386, 374)
NOSE_TIP = 1
MOUTH_LEFT = 61
MOUTH_RIGHT = 291
CHIN = 152


def _eye_center(pts: np.ndarray, indices: tuple[int, ...]) -> np.ndarray:
    return pts[list(indices)].mean(axis=0)


def _patch_points(lm: FaceLandmarks, offset: tuple[int, int]) -> np.ndarray:
    pts = lm.points.copy()
    pts[:, 0] -= offset[0]
    pts[:, 1] -= offset[1]
    return pts


def _similarity_from_landmarks(
    src_pts: np.ndarray,
    dst_pts: np.ndarray,
    *,
    max_angle_deg: float = 12.0,
    scale_range: tuple[float, float] = (0.82, 1.22),
) -> np.ndarray:
    """Constrained similarity transform using eye-line angle + inter-ocular scale."""
    le, re = _eye_center(src_pts, LEFT_EYE), _eye_center(src_pts, RIGHT_EYE)
    le_d, re_d = _eye_center(dst_pts, LEFT_EYE), _eye_center(dst_pts, RIGHT_EYE)

    src_dist = np.linalg.norm(re - le) + 1e-6
    dst_dist = np.linalg.norm(re_d - le_d)
    scale = np.clip(dst_dist / src_dist, scale_range[0], scale_range[1])

    src_ang = np.arctan2(re[1] - le[1], re[0] - le[0])
    dst_ang = np.arctan2(re_d[1] - le_d[1], re_d[0] - le_d[0])
    angle = np.clip(dst_ang - src_ang, -np.radians(max_angle_deg), np.radians(max_angle_deg))

    src_mid = (le + re) / 2
    dst_mid = (le_d + re_d) / 2

    cos_a, sin_a = np.cos(angle), np.sin(angle)
    rot_scale = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float64) * scale
    m = np.zeros((2, 3), dtype=np.float64)
    m[:, :2] = rot_scale
    m[:, 2] = dst_mid - rot_scale @ src_mid
    return m.astype(np.float32)


def estimate_similarity_transform(
    src: FaceLandmarks,
    dst: FaceLandmarks,
    src_offset: tuple[int, int] = (0, 0),
    dst_offset: tuple[int, int] = (0, 0),
) -> np.ndarray:
    src_pts = _patch_points(src, src_offset)
    dst_pts = _patch_points(dst, dst_offset)
    return _similarity_from_landmarks(src_pts, dst_pts)


def warp_patch(
    patch_bgr: np.ndarray,
    matrix: np.ndarray,
    output_size: tuple[int, int],
) -> np.ndarray:
    w, h = output_size
    return cv2.warpAffine(
        patch_bgr,
        matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
