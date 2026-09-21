"""Affine alignment of LOCK face patch onto target pose."""

from __future__ import annotations

import cv2
import numpy as np

from landmarks import (
    FaceLandmarks,
    LEFT_EYE_IDX,
    NOSE_TIP_IDX,
    RIGHT_EYE_IDX,
    align_points,
)

LEFT_EYE = LEFT_EYE_IDX
RIGHT_EYE = RIGHT_EYE_IDX
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


def _frontal_anchor_points(pts: np.ndarray) -> np.ndarray:
    """Five stable anchors: eye centers, nose, mouth corners."""
    le = _eye_center(pts, LEFT_EYE)
    re = _eye_center(pts, RIGHT_EYE)
    return np.array(
        [
            le,
            re,
            pts[NOSE_TIP],
            pts[MOUTH_LEFT],
            pts[MOUTH_RIGHT],
        ],
        dtype=np.float32,
    )


def estimate_frontal_transform(
    src: FaceLandmarks,
    dst: FaceLandmarks,
    src_offset: tuple[int, int] = (0, 0),
    dst_offset: tuple[int, int] = (0, 0),
    *,
    scale_range: tuple[float, float] = (0.45, 1.55),
    max_angle_deg: float = 8.0,
) -> np.ndarray:
    """Frontal identity: similarity fit on 5 anchors, minimal rotation."""
    src_pts = _patch_points(src, src_offset)
    dst_pts = _patch_points(dst, dst_offset)
    src_anchors = _frontal_anchor_points(src_pts)
    dst_anchors = _frontal_anchor_points(dst_pts)

    matrix, _ = cv2.estimateAffinePartial2D(
        src_anchors,
        dst_anchors,
        method=cv2.LMEDS,
    )
    if matrix is None:
        matrix = cv2.getAffineTransform(src_anchors[:3], dst_anchors[:3])

    le, re = _eye_center(src_pts, LEFT_EYE), _eye_center(src_pts, RIGHT_EYE)
    le_d, re_d = _eye_center(dst_pts, LEFT_EYE), _eye_center(dst_pts, RIGHT_EYE)
    src_dist = float(np.linalg.norm(re - le)) + 1e-6
    dst_dist = float(np.linalg.norm(re_d - le_d))
    raw_scale = dst_dist / src_dist

    a, b = float(matrix[0, 0]), float(matrix[0, 1])
    cur_scale = float(np.hypot(a, b))
    clamped = float(np.clip(raw_scale, scale_range[0], scale_range[1]))
    if cur_scale > 1e-6:
        ratio = clamped / cur_scale
        matrix = matrix.copy()
        matrix[:, :2] *= ratio

    src_ang = float(np.arctan2(re[1] - le[1], re[0] - le[0]))
    dst_ang = float(np.arctan2(re_d[1] - le_d[1], re_d[0] - le_d[0]))
    angle = float(np.clip(dst_ang - src_ang, -np.radians(max_angle_deg), np.radians(max_angle_deg)))
    cur_ang = float(np.arctan2(matrix[1, 0], matrix[0, 0]))
    delta = angle - cur_ang
    cos_d, sin_d = np.cos(delta), np.sin(delta)
    rot = np.array([[cos_d, -sin_d], [sin_d, cos_d]], dtype=np.float64)
    matrix[:, :2] = (rot @ matrix[:, :2].astype(np.float64)).astype(np.float32)

    src_mid = (le + re) / 2
    dst_mid = (le_d + re_d) / 2
    matrix = matrix.astype(np.float64)
    matrix[:, 2] = dst_mid - matrix[:, :2] @ src_mid
    return matrix.astype(np.float32)


def alignment_metrics(
    matrix: np.ndarray,
    src: FaceLandmarks,
    dst: FaceLandmarks,
    src_offset: tuple[int, int] = (0, 0),
    dst_offset: tuple[int, int] = (0, 0),
) -> dict[str, float]:
    """Diagnostics for warp sanity checks."""
    src_pts = _patch_points(src, src_offset)
    dst_pts = _patch_points(dst, dst_offset)
    le, re = _eye_center(src_pts, LEFT_EYE), _eye_center(src_pts, RIGHT_EYE)
    le_d, re_d = _eye_center(dst_pts, LEFT_EYE), _eye_center(dst_pts, RIGHT_EYE)
    src_dist = float(np.linalg.norm(re - le)) + 1e-6
    dst_dist = float(np.linalg.norm(re_d - le_d))
    a, b = float(matrix[0, 0]), float(matrix[0, 1])
    c, d = float(matrix[1, 0]), float(matrix[1, 1])
    return {
        "scale_x": float(np.hypot(a, b)),
        "scale_y": float(np.hypot(c, d)),
        "raw_iod_ratio": dst_dist / src_dist,
        "angle_deg": float(np.degrees(np.arctan2(matrix[1, 0], matrix[0, 0]))),
    }


def estimate_similarity_transform(
    src: FaceLandmarks,
    dst: FaceLandmarks,
    src_offset: tuple[int, int] = (0, 0),
    dst_offset: tuple[int, int] = (0, 0),
    *,
    frontal: bool = False,
) -> np.ndarray:
    if frontal:
        return estimate_frontal_transform(src, dst, src_offset, dst_offset)
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
