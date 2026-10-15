"""
Regenerate flagged master/yuna slots with pmt_yuna LoRA v3, archive originals, verify folders.

Usage (pickmetalk-lora venv):
  C:\\AI\\pickmetalk-lora\\venv\\Scripts\\python.exe scripts/regen-master-yuna-lora.py
"""
from __future__ import annotations

import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import cv2
import numpy as np
import torch
from diffusers import StableDiffusionXLPipeline
from insightface.app import FaceAnalysis

BASE_CKPT = Path(r"C:\AI\pickmetalk-lora\models\RealVisXL_V5.0_fp16.safetensors")
LORA = Path(r"D:\PickMeTalk_PhotoLibrary\loras\yuna\output\pmt_yuna_lora_v3.safetensors")
DATASET = Path(r"D:\PickMeTalk_PhotoLibrary\loras\yuna\dataset_v3\8_pmt_yuna")
MASTER = Path(r"D:\PickMeTalk_PhotoLibrary\master\yuna")
ARCHIVE_ROOT = MASTER / "_archive_identity_review" / datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")

FACE = (
    "pmt_yuna, close portrait of the same young Korean woman Yuna, head and shoulders, "
    "long straight dark brown hair with wispy see-through bangs, warm brown eyes, "
    "coral lip tint, natural skin pores, one person only, face large in frame"
)
NEG = (
    "lowres, blurry, deformed, bad anatomy, extra fingers, watermark, text, "
    "cartoon, anime, doll, plastic skin, wax, mannequin, different person, "
    "wrong face, face swap, uncanny, beauty filter airbrush"
)
W, H, STEPS, CFG, SCALE = 768, 1024, 24, 5.0, 0.9
EXTS = {".png", ".jpg", ".jpeg", ".webp"}
CENTROID_PASS = 0.40
MAX_PASS = 0.43

JOBS = [
    {
        "rel_dir": "05_outing/walk",
        "file": "02.png",
        "seed": 302,
        "prompt": (
            f"{FACE}, walking toward camera on a paved park path, trees and green grass, "
            "bright natural daylight, white ribbed tank top, relaxed gentle smile, "
            "front three-quarter full upper body, not looking over shoulder"
        ),
    },
    {
        "rel_dir": "11_emotions/angry",
        "file": "01.png",
        "seed": 401,
        "prompt": (
            f"{FACE}, clearly annoyed irritated expression, furrowed eyebrows, "
            "pressed lips no smile, slight glare at camera, off-shoulder knit top, "
            "warm sunset backlight, emotional but photorealistic"
        ),
    },
    {
        "rel_dir": "11_emotions/angry",
        "file": "02.png",
        "seed": 402,
        "prompt": (
            f"{FACE}, frustrated pout, eyebrows drawn together, lips tight, "
            "indoor soft light, simple beige knit, front-facing bust portrait"
        ),
    },
    {
        "rel_dir": "11_emotions/angry",
        "file": "03.png",
        "seed": 403,
        "prompt": (
            f"{FACE}, mild anger side glance, one eyebrow slightly raised, "
            "jaw set, no teeth visible, cafe background bokeh, cardigan"
        ),
    },
]

VERIFY_FOLDERS = ["05_outing/walk", "11_emotions/angry"]


def embed(app: FaceAnalysis, path: Path) -> np.ndarray | None:
    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return None
    faces = app.get(img)
    if not faces:
        pad_h, pad_w = img.shape[0] // 2, img.shape[1] // 2
        canvas = np.full((img.shape[0] + pad_h * 2, img.shape[1] + pad_w * 2, 3), 180, np.uint8)
        canvas[pad_h : pad_h + img.shape[0], pad_w : pad_w + img.shape[1]] = img
        faces = app.get(canvas)
    if not faces:
        return None
    face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    vec = face.normed_embedding.astype(np.float32)
    return vec / (np.linalg.norm(vec) + 1e-8)


def archive_file(src: Path, rel_dir: str) -> Path | None:
    if not src.is_file():
        return None
    dest_dir = ARCHIVE_ROOT / rel_dir.replace("/", os.sep)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / src.name
    shutil.move(str(src), str(dest))
    print(f"ARCHIVED {src} -> {dest}", flush=True)
    return dest


def load_pipe() -> StableDiffusionXLPipeline:
    print(f"LOAD {BASE_CKPT}", flush=True)
    pipe = StableDiffusionXLPipeline.from_single_file(
        str(BASE_CKPT), torch_dtype=torch.float16, use_safetensors=True
    )
    pipe = pipe.to("cuda")
    pipe.enable_vae_slicing()
    print(f"LORA {LORA} scale={SCALE}", flush=True)
    pipe.load_lora_weights(str(LORA.parent), weight_name=LORA.name)
    pipe.fuse_lora(lora_scale=SCALE)
    return pipe


def generate_one(pipe: StableDiffusionXLPipeline, job: dict, out: Path) -> None:
    g = torch.Generator(device="cuda").manual_seed(int(job["seed"]))
    print(f"GEN {out} seed={job['seed']}", flush=True)
    img = pipe(
        prompt=job["prompt"],
        negative_prompt=NEG,
        num_inference_steps=STEPS,
        guidance_scale=CFG,
        width=W,
        height=H,
        generator=g,
    ).images[0]
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out)
    print(f"WROTE {out} ({out.stat().st_size} bytes)", flush=True)


def build_ref_matrix(app: FaceAnalysis) -> tuple[np.ndarray, np.ndarray]:
    refs = []
    for p in sorted(DATASET.iterdir()):
        if p.suffix.lower() in EXTS:
            v = embed(app, p)
            if v is not None:
                refs.append(v)
    if len(refs) < 3:
        raise RuntimeError(f"Need >=3 dataset refs, got {len(refs)}")
    mat = np.stack(refs)
    centroid = mat.mean(0)
    centroid = centroid / (np.linalg.norm(centroid) + 1e-8)
    return mat, centroid


def score_image(app: FaceAnalysis, mat: np.ndarray, centroid: np.ndarray, path: Path) -> dict:
    v = embed(app, path)
    if v is None:
        return {"file": path.name, "ok": False, "reason": "no_face"}
    scores = mat @ v
    c = float(centroid @ v)
    mx = float(scores.max())
    ok = c >= CENTROID_PASS and mx >= MAX_PASS
    return {
        "file": path.name,
        "centroid": round(c, 4),
        "max": round(mx, 4),
        "ok": ok,
    }


def verify_folder(app: FaceAnalysis, mat: np.ndarray, centroid: np.ndarray, rel_dir: str) -> dict:
    folder = MASTER / rel_dir.replace("/", os.sep)
    files = sorted(folder.glob("*.png")) + sorted(folder.glob("*.jpg"))
    numbered = [p for p in files if p.name[:2].isdigit()]
    numbered = sorted(numbered, key=lambda p: p.name)[:3]
    rows = [score_image(app, mat, centroid, p) for p in numbered]
    folder_ok = len(rows) == 3 and all(r.get("ok") for r in rows)
    return {"folder": rel_dir, "count": len(rows), "folder_ok": folder_ok, "images": rows}


def main() -> int:
    if not LORA.is_file():
        print(f"MISSING LoRA {LORA}")
        return 2
    ARCHIVE_ROOT.mkdir(parents=True, exist_ok=True)
    report = {"archive_root": str(ARCHIVE_ROOT), "generated": [], "verify": []}

    pipe = load_pipe()
    for job in JOBS:
        out = MASTER / job["rel_dir"].replace("/", os.sep) / job["file"]
        if out.is_file():
            archive_file(out, job["rel_dir"])
        generate_one(pipe, job, out)
        report["generated"].append(str(out))
    del pipe
    torch.cuda.empty_cache()

    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=(640, 640))
    mat, centroid = build_ref_matrix(app)
    print(f"REFS {mat.shape[0]} centroid ready", flush=True)

    for rel in VERIFY_FOLDERS:
        v = verify_folder(app, mat, centroid, rel)
        report["verify"].append(v)
        status = "PASS" if v["folder_ok"] else "FAIL"
        print(f"VERIFY {rel} {status} {v}", flush=True)

    report_path = ARCHIVE_ROOT / "_regen_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"REPORT {report_path}", flush=True)
    all_ok = all(v["folder_ok"] for v in report["verify"])
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
