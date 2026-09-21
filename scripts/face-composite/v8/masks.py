"""Region masks for v8 layered compositing."""

from __future__ import annotations

import cv2
import numpy as np

from landmarks import FaceLandmarks, oval_points


def _normalize01(mask: np.ndarray) -> np.ndarray:
    m = mask.astype(np.float32)
    if m.max() > 1.0:
        m /= 255.0
    return np.clip(m, 0.0, 1.0)


def face_patch_bbox(
    landmarks: FaceLandmarks,
    image_shape: tuple[int, int],
    margin_ratio: float = 0.45,
) -> tuple[int, int, int, int]:
    """Return x, y, w, h for LOCK face patch crop (excludes most hair)."""
    h, w = image_shape[:2]
    pts = oval_points(landmarks)
    x_min, y_min = pts.min(axis=0)
    x_max, y_max = pts.max(axis=0)
    bw = x_max - x_min
    bh = y_max - y_min
    mx = bw * margin_ratio
    my = bh * margin_ratio
    x0 = int(max(0, x_min - mx))
    y0 = int(max(0, y_min - my * 1.1))  # less margin above (hair stays on target)
    x1 = int(min(w, x_max + mx))
    y1 = int(min(h, y_max + my * 1.6))  # more margin below for jaw-neck blend
    return x0, y0, x1 - x0, y1 - y0


def face_oval_mask(
    landmarks: FaceLandmarks,
    shape: tuple[int, int],
    offset: tuple[int, int] = (0, 0),
    *,
    expand: float = 1.06,
) -> np.ndarray:
    """Smooth ellipse mask (avoids polygonal hexagon seams from convex hull)."""
    h, w = shape[:2]
    mask = np.zeros((h, w), dtype=np.float32)
    pts = oval_points(landmarks).copy()
    pts[:, 0] -= offset[0]
    pts[:, 1] -= offset[1]
    if len(pts) >= 5:
        (cx, cy), (ax, ay), angle = cv2.fitEllipse(pts.astype(np.float32))
        ax, ay = ax * expand, ay * expand
        cv2.ellipse(mask, (int(cx), int(cy)), (int(ax / 2), int(ay / 2)), angle, 0, 360, 1.0, -1)
    else:
        cv2.fillConvexPoly(mask, pts.astype(np.int32), 1.0)
    return mask


def distance_feather_mask(
    binary: np.ndarray,
    inner_px: int = 8,
    outer_px: int = 36,
) -> np.ndarray:
    """Smooth alpha from distance to mask boundary (no rectangular artifacts)."""
    bin_u8 = (binary > 0.5).astype(np.uint8)
    dist_in = cv2.distanceTransform(bin_u8, cv2.DIST_L2, 5)
    dist_out = cv2.distanceTransform(1 - bin_u8, cv2.DIST_L2, 5)
    inside = np.clip(dist_in / max(inner_px, 1), 0.0, 1.0)
    outside = np.clip(1.0 - dist_out / max(outer_px, 1), 0.0, 1.0)
    alpha = np.where(bin_u8 > 0, inside, outside)
    return np.clip(alpha, 0.0, 1.0)


def region_masks(
    landmarks: FaceLandmarks,
    shape: tuple[int, int],
    offset: tuple[int, int] = (0, 0),
    *,
    inner_px: int = 14,
    outer_px: int = 58,
    jaw_extend_px: int = 56,
) -> dict[str, np.ndarray]:
    h, w = shape[:2]
    oval = face_oval_mask(landmarks, shape, offset)
    pts = oval_points(landmarks).copy()
    pts[:, 0] -= offset[0]
    pts[:, 1] -= offset[1]

    # Extend oval below chin for jaw-neck transition.
    chin_y = int(pts[:, 1].max())
    jaw_band = np.zeros((h, w), dtype=np.float32)
    extend_end = min(h, chin_y + jaw_extend_px)
    for y in range(chin_y, extend_end):
        t = (y - chin_y) / max(jaw_extend_px, 1)
        jaw_band[y, :] = max(0.0, 1.0 - t * 0.15)

    oval_extended = np.clip(oval + jaw_band * 0.25, 0.0, 1.0)
    alpha = distance_feather_mask(oval_extended > 0.35, inner_px=inner_px, outer_px=outer_px)

    # Forehead fade: eyebrow line (~ landmark 10 top) upward.
    brow_y = int(np.percentile(pts[:, 1], 12))
    forehead = np.ones((h, w), dtype=np.float32)
    fade_h = int((chin_y - brow_y) * 0.28)
    for y in range(max(0, brow_y - fade_h), brow_y):
        t = (brow_y - y) / max(fade_h, 1)
        forehead[y, :] = 1.0 - t * 0.88
    alpha *= forehead
    alpha = cv2.GaussianBlur(alpha, (0, 0), 4.5)

    core = cv2.erode(
        (oval > 0.5).astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (inner_px * 2 + 1, inner_px * 2 + 1)),
    ).astype(np.float32)
    transition = np.clip(alpha - core, 0.0, 1.0)

    return {
        "oval": oval,
        "core": core,
        "transition": transition,
        "forehead_fade": 1.0 - forehead,
        "jaw_neck": jaw_band,
        "alpha": alpha,
    }


def region_masks_frontal(
    landmarks: FaceLandmarks,
    shape: tuple[int, int],
    offset: tuple[int, int] = (0, 0),
    *,
    outer_px: int = 20,
) -> dict[str, np.ndarray]:
    """
    Frontal identity mode: LOCK pixels fill almost entire face oval.
    Feather only at the outer rim (jaw/cheek), no forehead fade-down.
    """
    h, w = shape[:2]
    oval = face_oval_mask(landmarks, shape, offset, expand=1.03)
    pts = oval_points(landmarks).copy()
    pts[:, 0] -= offset[0]
    pts[:, 1] -= offset[1]

    chin_y = int(pts[:, 1].max())
    jaw_band = np.zeros((h, w), dtype=np.float32)
    extend_end = min(h, chin_y + 36)
    for y in range(chin_y, extend_end):
        t = (y - chin_y) / max(36, 1)
        jaw_band[y, :] = max(0.0, 1.0 - t * 0.2)

    alpha = distance_feather_mask(oval > 0.45, inner_px=32, outer_px=outer_px)
    alpha = np.clip(alpha + jaw_band * 0.08, 0.0, 1.0)

    core = cv2.erode(
        (oval > 0.5).astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    ).astype(np.float32)
    # Core = 100% LOCK pixels (identity identical).
    alpha = np.clip(np.maximum(alpha, core), 0.0, 1.0)
    alpha = cv2.GaussianBlur(alpha, (0, 0), 1.2)
    alpha = np.where(core > 0.5, 1.0, alpha)
    transition = np.clip(alpha - core, 0.0, 1.0)

    return {
        "oval": oval,
        "core": core,
        "transition": transition,
        "forehead_fade": np.zeros((h, w), dtype=np.float32),
        "jaw_neck": jaw_band,
        "alpha": alpha,
    }


def refine_alpha_with_target_skin(
    alpha: np.ndarray,
    target_skin_mask: np.ndarray,
    *,
    blend_strength: float = 0.25,
) -> np.ndarray:
    skin = _normalize01(target_skin_mask)
    refined = alpha * (1.0 - blend_strength) + alpha * skin * blend_strength
    return np.clip(refined, 0.0, 1.0)


def fringe_hair_mask(
    hair_mask: np.ndarray,
    landmarks: FaceLandmarks,
    offset: tuple[int, int] = (0, 0),
) -> np.ndarray:
    """Keep target hair only in fringe / above-eyebrow zones for natural bangs."""
    h, w = hair_mask.shape[:2]
    pts = oval_points(landmarks).copy()
    pts[:, 0] -= offset[0]
    pts[:, 1] -= offset[1]
    brow_y = int(np.percentile(pts[:, 1], 15))
    fringe_zone = np.zeros((h, w), dtype=np.float32)
    fringe_zone[:brow_y + 12, :] = 1.0
    # Also allow hair at sides overlapping cheeks.
    x_min, x_max = int(pts[:, 0].min()), int(pts[:, 0].max())
    side = int((x_max - x_min) * 0.12)
    fringe_zone[:, : x_min + side] = 1.0
    fringe_zone[:, x_max - side :] = 1.0
    return np.clip(hair_mask * fringe_zone, 0.0, 1.0)
