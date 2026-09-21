"""Hair and selfie multiclass segmentation."""

from __future__ import annotations

from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# selfie_multiclass categories
SELFIE_BG = 0
SELFIE_HAIR = 1
SELFIE_BODY_SKIN = 2
SELFIE_FACE_SKIN = 3
SELFIE_CLOTHES = 4
SELFIE_OTHERS = 5


class ImageSegmenterService:
    def __init__(self, model_path: Path) -> None:
        options = vision.ImageSegmenterOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
            output_category_mask=True,
            output_confidence_masks=True,
        )
        self._segmenter = vision.ImageSegmenter.create_from_options(options)

    def segment(self, image_bgr: np.ndarray) -> tuple[np.ndarray, list[np.ndarray] | None]:
        h, w = image_bgr.shape[:2]
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._segmenter.segment(mp_image)
        category = result.category_mask.numpy_view().squeeze().astype(np.uint8)
        if category.shape[:2] != (h, w):
            category = cv2.resize(category, (w, h), interpolation=cv2.INTER_NEAREST)
        conf = None
        if result.confidence_masks:
            conf = []
            for mask in result.confidence_masks:
                arr = mask.numpy_view().squeeze().astype(np.float32)
                if arr.shape[:2] != (h, w):
                    arr = cv2.resize(arr, (w, h), interpolation=cv2.INTER_LINEAR)
                conf.append(arr)
        return category, conf

    def close(self) -> None:
        self._segmenter.close()


def hair_mask_from_segmentation(
    category: np.ndarray,
    confidence: list[np.ndarray] | None,
    *,
    use_multiclass: bool,
) -> np.ndarray:
    """Return float32 hair mask in [0, 1]."""
    if use_multiclass:
        if confidence and len(confidence) > SELFIE_HAIR:
            return np.clip(confidence[SELFIE_HAIR], 0.0, 1.0).astype(np.float32)
        return (category == SELFIE_HAIR).astype(np.float32)
    if confidence and len(confidence) > 1:
        return np.clip(confidence[1], 0.0, 1.0).astype(np.float32)
    return (category == 1).astype(np.float32)


def face_skin_mask_from_segmentation(
    category: np.ndarray,
    confidence: list[np.ndarray] | None,
) -> np.ndarray:
    if confidence and len(confidence) > SELFIE_FACE_SKIN:
        skin = np.clip(confidence[SELFIE_FACE_SKIN], 0.0, 1.0)
        if len(confidence) > SELFIE_BODY_SKIN:
            skin = np.maximum(skin, np.clip(confidence[SELFIE_BODY_SKIN], 0.0, 1.0))
        return skin.astype(np.float32)
    return ((category == SELFIE_FACE_SKIN) | (category == SELFIE_BODY_SKIN)).astype(np.float32)
