#!/usr/bin/env python3
"""
v8 LOCK face compositor — identity via pixel lock, natural seams via layered masks.

Frontal mode (--frontal / auto): 정면 얼굴은 LOCK 픽셀을 최대한 동일하게 유지.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from align import alignment_metrics, estimate_similarity_transform, warp_patch
from blend import (
    composite_layers,
    frequency_blend,
    harmonize_frontal_identity,
    lab_transfer_zone,
    laplacian_blend,
)
from landmarks import FaceLandmarkerService, is_frontal, pose_metrics
from masks import (
    face_patch_bbox,
    face_patch_bbox_frontal,
    fringe_hair_mask,
    region_masks,
    region_masks_frontal,
)
from segmentation import (
    ImageSegmenterService,
    face_skin_mask_from_segmentation,
    hair_mask_from_segmentation,
)

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def _read_bgr(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return img


def _write_debug(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)


def _mask_preview(mask: np.ndarray) -> np.ndarray:
    m = np.clip(mask * 255.0, 0, 255).astype(np.uint8)
    return cv2.applyColorMap(m, cv2.COLORMAP_VIRIDIS)


def run_composite(
    lock_path: Path,
    target_path: Path,
    output_path: Path,
    *,
    debug_dir: Path | None = None,
    color_match: float = 0.55,
    hair_overlay_strength: float = 0.95,
    laplacian: bool = False,
    frontal: str = "auto",
    identity_boost: bool = False,
) -> dict:
    lock_bgr = _read_bgr(lock_path)
    target_bgr = _read_bgr(target_path)
    th, tw = target_bgr.shape[:2]

    landmarker = FaceLandmarkerService(MODELS_DIR / "face_landmarker.task")
    selfie_segmenter = ImageSegmenterService(MODELS_DIR / "selfie_multiclass.tflite")
    hair_segmenter = ImageSegmenterService(MODELS_DIR / "hair_segmenter.tflite")

    try:
        lock_lm = landmarker.detect(lock_bgr)
        target_lm = landmarker.detect(target_bgr)
        if lock_lm is None:
            raise RuntimeError(f"No face detected in LOCK image: {lock_path}")
        if target_lm is None:
            raise RuntimeError(f"No face detected in target image: {target_path}")

        lock_frontal = is_frontal(lock_lm)
        target_frontal = is_frontal(target_lm)
        tgt_tilt, _ = pose_metrics(target_lm)
        near_frontal = abs(tgt_tilt) <= 32.0
        use_frontal = (
            frontal == "on"
            or (frontal == "auto" and lock_frontal and (target_frontal or near_frontal))
        )

        if use_frontal:
            lx, ly, lw, lh = face_patch_bbox_frontal(lock_lm, lock_bgr.shape)
        else:
            lx, ly, lw, lh = face_patch_bbox(lock_lm, lock_bgr.shape)
        lock_patch = lock_bgr[ly : ly + lh, lx : lx + lw].copy()

        matrix = estimate_similarity_transform(
            lock_lm, target_lm, src_offset=(lx, ly), dst_offset=(0, 0), frontal=use_frontal
        )
        align_meta = alignment_metrics(matrix, lock_lm, target_lm, (lx, ly), (0, 0))
        lock_warped = warp_patch(lock_patch, matrix, (tw, th))

        if use_frontal:
            zones = region_masks_frontal(target_lm, (th, tw))
            alpha = zones["alpha"]
        else:
            zones = region_masks(target_lm, (th, tw))
            alpha = zones["alpha"]
            target_cat, target_conf = selfie_segmenter.segment(target_bgr)
            target_skin = face_skin_mask_from_segmentation(target_cat, target_conf)
            from masks import refine_alpha_with_target_skin
            alpha = refine_alpha_with_target_skin(alpha, target_skin)

        target_cat, target_conf = selfie_segmenter.segment(target_bgr)
        hair_cat, hair_conf = hair_segmenter.segment(target_bgr)
        hair_mask = hair_mask_from_segmentation(hair_cat, hair_conf, use_multiclass=False)
        mc_hair = hair_mask_from_segmentation(target_cat, target_conf, use_multiclass=True)
        hair_mask = np.clip(np.maximum(hair_mask, mc_hair), 0.0, 1.0)
        hair_mask = fringe_hair_mask(hair_mask, target_lm)
        if use_frontal:
            from landmarks import oval_points
            pts = oval_points(target_lm)
            brow_y = int(np.percentile(pts[:, 1], 12))
            yy = np.arange(th, dtype=np.float32)[:, None]
            hair_mask *= (yy < brow_y + 6).astype(np.float32)
        hair_mask = cv2.GaussianBlur(hair_mask, (5, 5), 1.0)
        strength = hair_overlay_strength * (0.4 if use_frontal else 1.0)
        hair_mask = np.clip(hair_mask * strength, 0.0, 1.0)

        valid = (lock_warped.sum(axis=2) > 12).astype(np.float32)
        alpha = np.clip(alpha * valid, 0.0, 1.0)
        zones["valid_warp"] = valid

        if use_frontal:
            lock_blended = lock_warped
        else:
            color_zone = np.clip(zones["transition"] + zones["jaw_neck"] * 0.85, 0.0, 1.0)
            lock_matched = lab_transfer_zone(
                lock_warped, target_bgr, color_zone, amount=color_match
            )
            blend_alpha = np.clip(
                zones["transition"] + zones["jaw_neck"] * 0.65 + zones["forehead_fade"] * 0.4,
                0.0,
                1.0,
            )
            lock_blended = frequency_blend(lock_matched, target_bgr, blend_alpha)

        if use_frontal:
            boost = identity_boost or (near_frontal and not target_frontal)
            pasted = harmonize_frontal_identity(
                target_bgr,
                lock_blended,
                zones,
                color_match=max(color_match, 0.65),
                core_luma=0.1 if not boost else 0.0,
                feather_px=20,
                poisson_rim=(target_frontal and not boost),
                identity_boost=boost,
            )
            result = composite_layers(target_bgr, pasted, alpha, hair_mask)
            valid_warp = zones.get("valid_warp", valid) > 0.5
            core = (zones["core"] > 0.5) & valid_warp & (hair_mask < 0.4)
            result[core] = pasted[core]
        elif laplacian:
            merged = laplacian_blend(lock_blended, target_bgr, alpha)
            result = composite_layers(target_bgr, merged, alpha, hair_mask)
        else:
            result = composite_layers(target_bgr, lock_blended, alpha, hair_mask)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), result)

        lock_tilt, lock_nose = pose_metrics(lock_lm)
        tgt_tilt, tgt_nose = pose_metrics(target_lm)

        meta = {
            "version": "v8-frontal" if use_frontal else "v8",
            "mode": "frontal" if use_frontal else "general",
            "lock": str(lock_path),
            "target": str(target_path),
            "output": str(output_path),
            "patch_bbox": {"x": lx, "y": ly, "w": lw, "h": lh},
            "pose": {
                "lock_frontal": lock_frontal,
                "target_frontal": target_frontal,
                "lock_tilt_deg": lock_tilt,
                "target_tilt_deg": tgt_tilt,
            },
            "alignment": align_meta,
            "size": {"width": tw, "height": th},
        }

        if debug_dir:
            debug_dir.mkdir(parents=True, exist_ok=True)
            _write_debug(debug_dir / "00_lock_patch.jpg", lock_patch)
            _write_debug(debug_dir / "01_lock_warped.jpg", lock_warped)
            _write_debug(debug_dir / "02_alpha.jpg", _mask_preview(alpha))
            _write_debug(debug_dir / "03_hair_overlay.jpg", _mask_preview(hair_mask))
            _write_debug(debug_dir / "05_result.jpg", result)
            with open(debug_dir / "meta.json", "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)

        return meta
    finally:
        landmarker.close()
        selfie_segmenter.close()
        hair_segmenter.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="v8 LOCK face compositor")
    parser.add_argument("--lock", required=True, help="LOCK PRIMARY image (01_front_main)")
    parser.add_argument("--target", required=True, help="Body/scene target image")
    parser.add_argument("--output", required=True, help="Output composite path")
    parser.add_argument("--debug-dir", default=None, help="Write mask/debug intermediates")
    parser.add_argument("--color-match", type=float, default=0.55)
    parser.add_argument("--hair-strength", type=float, default=0.95)
    parser.add_argument("--frontal", choices=["auto", "on", "off"], default="auto")
    parser.add_argument("--no-laplacian", action="store_true")
    args = parser.parse_args(argv)

    meta = run_composite(
        Path(args.lock),
        Path(args.target),
        Path(args.output),
        debug_dir=Path(args.debug_dir) if args.debug_dir else None,
        color_match=args.color_match,
        hair_overlay_strength=args.hair_strength,
        laplacian=not args.no_laplacian,
        frontal=args.frontal,
    )
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
