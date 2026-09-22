#!/usr/bin/env python3
"""Compare LoRA samples to Midjourney base faces with ArcFace cosine similarity.

Read-only. This does not edit pixels and does not run a face-swap model.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def largest_embedding(app, image_path: Path) -> np.ndarray | None:
    import cv2

    image = cv2.imread(str(image_path))
    if image is None:
        return None
    faces = app.get(image)
    if not faces:
        return None
    face = max(faces, key=lambda item: (item.bbox[2] - item.bbox[0]) * (item.bbox[3] - item.bbox[1]))
    return np.asarray(face.normed_embedding, dtype=np.float32)


def main() -> int:
    parser = argparse.ArgumentParser(description="ArcFace identity check")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    if manifest.get("method") != "lora-txt2img" or manifest.get("compositing") is not False:
        print("refusing identity check: method is not lora-txt2img", file=sys.stderr)
        return 2

    samples_dir = Path(manifest["outputDir"]) / "samples"
    sample_paths = sorted(
        p for p in samples_dir.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    )
    base_paths = [Path(item["path"]) for item in manifest.get("baseImages") or []]
    report_path = Path(args.out)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    if not sample_paths or not base_paths:
        report = {"engine": "none", "error": "samples_or_base_missing", "pairs": []}
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print("samples or base images are missing", file=sys.stderr)
        return 2

    try:
        from insightface.app import FaceAnalysis
    except ImportError:
        report = {"engine": "none", "error": "insightface_not_installed", "pairs": []}
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print("insightface is not installed; identity was not confirmed", file=sys.stderr)
        return 4

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    app = FaceAnalysis(name="buffalo_l", providers=providers)
    ctx = 0
    try:
        import torch

        ctx = 0 if torch.cuda.is_available() else -1
    except ImportError:
        ctx = -1
    app.prepare(ctx_id=ctx, det_size=(640, 640))

    base_embeddings = []
    for path in base_paths:
        embedding = largest_embedding(app, path)
        if embedding is not None:
            base_embeddings.append((str(path), embedding))
    if not base_embeddings:
        report = {"engine": "arcface", "error": "no_face_in_base", "pairs": []}
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return 4

    pairs = []
    for sample in sample_paths:
        embedding = largest_embedding(app, sample)
        if embedding is None:
            pairs.append({"sample": sample.name, "similarity": 0.0, "face": False})
            continue
        best = max(cosine(embedding, base) for _, base in base_embeddings)
        pairs.append({"sample": sample.name, "similarity": best, "face": True})

    scores = [pair["similarity"] for pair in pairs]
    report = {
        "engine": "arcface",
        "compositing": False,
        "method": "lora-txt2img",
        "pairs": pairs,
        "meanSimilarity": sum(scores) / len(scores) if scores else None,
        "minSimilarity": min(scores) if scores else None,
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"meanSimilarity": report["meanSimilarity"], "minSimilarity": report["minSimilarity"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
