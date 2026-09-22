#!/usr/bin/env python3
"""Sample a character LoRA with text prompts only.

Image-conditioned generation is refused. Nothing in this file reads a face
image as an init frame or as a conditioning embedding.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def refuse(message: str) -> int:
    print(message, file=sys.stderr)
    return 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LoRA text-to-image samples")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--init-image", default=None)
    parser.add_argument("--method", default="lora-txt2img")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.init_image or args.method != "lora-txt2img":
        return refuse("refusing image-conditioned generation; LoRA text-to-image only")

    manifest_path = Path(args.manifest)
    if not manifest_path.is_file():
        return refuse(f"manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("compositing") is not False or manifest.get("method") != "lora-txt2img":
        return refuse("refusing to sample: manifest is not lora-txt2img without compositing")

    adapter_dir = Path(manifest["outputDir"]) / "adapter"
    method_path = adapter_dir / "method.json"
    if method_path.is_file():
        method = json.loads(method_path.read_text(encoding="utf-8"))
        if method.get("compositing") is not False or method.get("method") != "lora-txt2img":
            return refuse("refusing to sample: adapter was not trained as lora-txt2img")

    import torch
    from diffusers import StableDiffusionXLPipeline

    if not torch.cuda.is_available():
        print("NVIDIA RTX CUDA device is required to sample.", file=sys.stderr)
        return 3
    name = torch.cuda.get_device_name(0)
    if "RTX" not in name.upper() and __import__("os").environ.get("LORA_ALLOW_NON_RTX") != "1":
        print(f"Expected an NVIDIA RTX GPU, found: {name}", file=sys.stderr)
        return 3

    train = manifest.get("train") or {}
    model_id = train.get("modelId") or "stabilityai/stable-diffusion-xl-base-1.0"
    pipe = StableDiffusionXLPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
    )
    pipe.load_lora_weights(str(adapter_dir))
    pipe.to("cuda")

    out_dir = Path(manifest["outputDir"]) / "samples"
    out_dir.mkdir(parents=True, exist_ok=True)
    prompts = list(manifest.get("prompts") or [])
    negative = manifest.get("negativePrompt") or ""
    written = []
    for index, prompt in enumerate(prompts, start=1):
        # Prompt only. No image argument is passed into the pipeline.
        image = pipe(
            prompt=prompt,
            negative_prompt=negative,
            num_inference_steps=28,
            guidance_scale=5.0,
            height=1024,
            width=768,
        ).images[0]
        dest = out_dir / f"sample_{index:02d}.png"
        image.save(dest)
        written.append({"file": dest.name, "prompt": prompt, "method": "lora-txt2img"})
        print(f"wrote {dest}")

    (out_dir / "samples.json").write_text(
        json.dumps({"method": "lora-txt2img", "compositing": False, "samples": written}, indent=2),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
