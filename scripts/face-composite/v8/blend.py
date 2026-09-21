"""Color matching and frequency-aware boundary blending."""

from __future__ import annotations

import cv2
import numpy as np


def _to_lab(image_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)


def _from_lab(lab: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)


def lab_transfer_zone(
    source_bgr: np.ndarray,
    target_bgr: np.ndarray,
    zone_mask: np.ndarray,
    *,
    amount: float = 0.65,
) -> np.ndarray:
    zone = zone_mask > 0.05
    if not np.any(zone):
        return source_bgr

    src_lab = _to_lab(source_bgr)
    tgt_lab = _to_lab(target_bgr)
    out_lab = src_lab.copy()
    blend = np.clip(zone_mask, 0.0, 1.0) * amount

    for c in range(3):
        s = src_lab[:, :, c][zone]
        t = tgt_lab[:, :, c][zone]
        s_mean, s_std = float(s.mean()), float(s.std() + 1e-6)
        t_mean, t_std = float(t.mean()), float(t.std() + 1e-6)
        channel = (src_lab[:, :, c] - s_mean) * (t_std / s_std) + t_mean
        out_lab[:, :, c] = src_lab[:, :, c] * (1.0 - blend) + channel * blend

    return _from_lab(out_lab)


def frequency_blend(
    lock_bgr: np.ndarray,
    target_bgr: np.ndarray,
    alpha: np.ndarray,
    *,
    low_sigma: float = 4.5,
) -> np.ndarray:
    a = np.clip(alpha, 0.0, 1.0)
    lock_f = lock_bgr.astype(np.float32)
    tgt_f = target_bgr.astype(np.float32)
    lock_low = cv2.GaussianBlur(lock_f, (0, 0), low_sigma)
    tgt_low = cv2.GaussianBlur(tgt_f, (0, 0), low_sigma)
    lock_high = lock_f - lock_low
    low_blend = lock_low * (1.0 - a[..., None]) + tgt_low * a[..., None]
    return np.clip(low_blend + lock_high, 0, 255).astype(np.uint8)


def poisson_refine(
    source_bgr: np.ndarray,
    target_bgr: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """Seamless clone in transition band — lighting only, identity pixels preserved."""
    m = (mask > 0.15).astype(np.uint8) * 255
    if cv2.countNonZero(m) < 100:
        return target_bgr
    ys, xs = np.where(m > 0)
    cx = int(xs.mean())
    cy = int(ys.mean())
    try:
        return cv2.seamlessClone(source_bgr, target_bgr, m, (cx, cy), cv2.MIXED_CLONE)
    except cv2.error:
        return target_bgr


def laplacian_blend(
    source_bgr: np.ndarray,
    target_bgr: np.ndarray,
    mask: np.ndarray,
    levels: int = 5,
) -> np.ndarray:
    """Multi-band blend for seamless transition (identity in high-freq from source)."""
    m = np.clip(mask, 0.0, 1.0).astype(np.float32)
    gp_m = [m]
    gp_s = [source_bgr.astype(np.float32)]
    gp_t = [target_bgr.astype(np.float32)]
    for _ in range(levels):
        gp_m.append(cv2.pyrDown(gp_m[-1]))
        gp_s.append(cv2.pyrDown(gp_s[-1]))
        gp_t.append(cv2.pyrDown(gp_t[-1]))

    lp_s = [gp_s[levels]]
    lp_t = [gp_t[levels]]
    for i in range(levels, 0, -1):
        size = (gp_s[i - 1].shape[1], gp_s[i - 1].shape[0])
        ls = cv2.pyrUp(lp_s[-1], dstsize=size)
        lt = cv2.pyrUp(lp_t[-1], dstsize=size)
        lp_s.append(gp_s[i - 1] - ls)
        lp_t.append(gp_t[i - 1] - lt)

    blended = []
    for i, (ls, lt, gm) in enumerate(zip(lp_s, lp_t, reversed(gp_m))):
        if gm.shape[:2] != ls.shape[:2]:
            gm = cv2.resize(gm, (ls.shape[1], ls.shape[0]))
        blended.append(ls * gm[..., None] + lt * (1.0 - gm[..., None]))

    out = blended[0]
    for i in range(1, levels + 1):
        size = (blended[i].shape[1], blended[i].shape[0])
        out = cv2.pyrUp(out, dstsize=size) + blended[i]
    return np.clip(out, 0, 255).astype(np.uint8)


def _luminance_adapt(
    source_bgr: np.ndarray,
    target_bgr: np.ndarray,
    mask: np.ndarray,
    *,
    amount: float = 0.22,
) -> np.ndarray:
    """Subtle L-channel mean match — keeps hue/identity, reduces lighting jump."""
    zone = mask > 0.45
    if not np.any(zone):
        return source_bgr
    src_lab = _to_lab(source_bgr)
    tgt_lab = _to_lab(target_bgr)
    out_lab = src_lab.copy()
    s_l = src_lab[:, :, 0][zone]
    t_l = tgt_lab[:, :, 0][zone]
    delta = float(t_l.mean() - s_l.mean()) * amount
    out_lab[:, :, 0] = np.clip(src_lab[:, :, 0] + delta * mask, 0, 255)
    return _from_lab(out_lab)


def _adaptive_core_luma(
    lock_bgr: np.ndarray,
    target_bgr: np.ndarray,
    core: np.ndarray,
) -> float:
    zone = core > 0.45
    if not np.any(zone):
        return 0.22
    lock_l = _to_lab(lock_bgr)[:, :, 0][zone]
    tgt_l = _to_lab(target_bgr)[:, :, 0][zone]
    delta = abs(float(tgt_l.mean() - lock_l.mean()))
    return float(np.clip(0.12 + delta / 120.0, 0.12, 0.48))


def poisson_rim_refine(
    composite_bgr: np.ndarray,
    target_bgr: np.ndarray,
    zones: dict[str, np.ndarray],
    valid: np.ndarray,
) -> np.ndarray:
    """Poisson clone on transition ring only — lighting seam, core untouched."""
    core = zones["core"] > 0.5
    transition = zones.get("transition", np.zeros_like(core, dtype=np.float32))
    rim = (transition > 0.12) & ~core & (valid > 0.5)
    rim_u8 = (rim.astype(np.uint8) * 255)
    if cv2.countNonZero(rim_u8) < 80:
        return composite_bgr
    ys, xs = np.where(rim_u8 > 0)
    cx, cy = int(xs.mean()), int(ys.mean())
    try:
        refined = cv2.seamlessClone(composite_bgr, target_bgr, rim_u8, (cx, cy), cv2.MIXED_CLONE)
    except cv2.error:
        return composite_bgr
    out = composite_bgr.copy()
    out[rim] = refined[rim]
    out[core & (valid > 0.5)] = composite_bgr[core & (valid > 0.5)]
    return out


def harmonize_frontal_identity(
    target_bgr: np.ndarray,
    lock_warped_bgr: np.ndarray,
    zones: dict[str, np.ndarray],
    *,
    color_match: float = 0.62,
    core_luma: float | None = None,
    feather_px: int = 22,
    poisson_rim: bool = True,
    identity_boost: bool = False,
) -> np.ndarray:
    """
    정면 identity: 코어는 LOCK 픽셀 유지 + 미세 밝기 맞춤, 링만 색상/주파수 블렌드.
    """
    oval = zones["oval"]
    core = zones["core"]
    transition = zones["transition"]
    h, w = oval.shape[:2]
    bin_u8 = (oval > 0.45).astype(np.uint8)
    if cv2.countNonZero(bin_u8) < 100:
        return target_bgr

    valid = (lock_warped_bgr.sum(axis=2) > 12).astype(np.float32)
    if identity_boost:
        luma = 0.0
        rim_color = min(color_match, 0.45)
        feather_px = min(feather_px, 14)
    else:
        luma = core_luma if core_luma is not None else _adaptive_core_luma(lock_warped_bgr, target_bgr, core)
        luma = min(luma, 0.18)
        rim_color = color_match
    lock_core = (
        lock_warped_bgr
        if luma <= 0.01
        else _luminance_adapt(lock_warped_bgr, target_bgr, core * valid, amount=luma)
    )

    ring_zone = np.clip(transition + zones.get("jaw_neck", 0) * 0.5, 0.0, 1.0)
    if identity_boost:
        lock_blend = lock_warped_bgr
    else:
        lock_matched = lab_transfer_zone(
            lock_warped_bgr,
            target_bgr,
            ring_zone,
            amount=rim_color,
        )
        lock_blend = frequency_blend(lock_matched, target_bgr, ring_zone, low_sigma=5.0)

    dist_in = cv2.distanceTransform(bin_u8, cv2.DIST_L2, 5)
    result = target_bgr.astype(np.float32)
    lock_f = lock_blend.astype(np.float32)
    core_f = lock_core.astype(np.float32)
    tgt_f = target_bgr.astype(np.float32)

    hard_core = (core > 0.5) & (valid > 0.5)
    result[hard_core] = core_f[hard_core]

    # Warp does not always cover the full oval (chin/jaw) — keep target pixels there.
    uncovered = (oval > 0.42) & (valid < 0.5)
    result[uncovered] = tgt_f[uncovered]

    hard_lock = (dist_in > feather_px) & (valid > 0.5) & ~hard_core
    result[hard_lock] = lock_f[hard_lock]

    ring = (dist_in > 0) & (dist_in <= feather_px) & (valid > 0.5)
    t = np.clip(dist_in / max(feather_px, 1), 0.0, 1.0)
    for c in range(3):
        ch = result[:, :, c]
        ch[ring] = lock_f[:, :, c][ring] * t[ring] + tgt_f[:, :, c][ring] * (1.0 - t[ring])
        result[:, :, c] = ch

    out = np.clip(result, 0, 255).astype(np.uint8)
    if poisson_rim:
        out = poisson_rim_refine(out, target_bgr, zones, valid)
    return out


def composite_layers(
    target_bgr: np.ndarray,
    lock_warped_bgr: np.ndarray,
    alpha: np.ndarray,
    hair_overlay_mask: np.ndarray,
) -> np.ndarray:
    a = np.clip(alpha, 0.0, 1.0)[..., None]
    base = target_bgr.astype(np.float32) * (1.0 - a) + lock_warped_bgr.astype(np.float32) * a
    hair = np.clip(hair_overlay_mask, 0.0, 1.0)[..., None]
    result = base * (1.0 - hair) + target_bgr.astype(np.float32) * hair
    return np.clip(result, 0, 255).astype(np.uint8)
