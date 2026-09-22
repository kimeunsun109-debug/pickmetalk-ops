#!/usr/bin/env python3
"""Train an SDXL character LoRA from Midjourney base faces.

Text-to-image LoRA only. This script does not paste, swap, or blend a face
onto another image. The base photos are reconstruction targets for the adapter.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def fail(message: str, code: int) -> int:
    print(message, file=sys.stderr)
    return code


def load_manifest(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("compositing") is not False or data.get("method") != "lora-txt2img":
        raise SystemExit(fail("refusing to train: manifest is not lora-txt2img without compositing", 2))
    return data


def assert_rtx() -> str:
    import torch

    if not torch.cuda.is_available():
        raise SystemExit(
            fail("NVIDIA RTX CUDA device is required. This machine has no CUDA GPU.", 3)
        )
    name = torch.cuda.get_device_name(0)
    allow = __import__("os").environ.get("LORA_ALLOW_NON_RTX") == "1"
    if "RTX" not in name.upper() and not allow:
        raise SystemExit(fail(f"Expected an NVIDIA RTX GPU, found: {name}", 3))
    return name


def main() -> int:
    parser = argparse.ArgumentParser(description="Train SDXL LoRA from a character base manifest")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--check", action="store_true", help="Validate manifest and RTX only")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.is_file():
        return fail(f"manifest not found: {manifest_path}", 2)
    manifest = load_manifest(manifest_path)
    if not manifest.get("readyToTrain"):
        return fail(
            f"need at least {manifest.get('minTrainImages')} base images, found {manifest.get('imageCount')}",
            2,
        )
    dataset_dir = Path(manifest["datasetDir"])
    images = [
        p
        for p in dataset_dir.glob("*")
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    ]
    if len(images) < int(manifest.get("minTrainImages") or 5):
        return fail(f"dataset is missing images: {dataset_dir}", 2)

    device_name = assert_rtx()
    print(f"RTX: {device_name}")
    if args.check:
        print("check ok: lora-txt2img, compositing disabled")
        return 0

    import numpy as np
    import torch
    from diffusers import DDPMScheduler, StableDiffusionXLPipeline
    from peft import LoraConfig
    from peft.utils import get_peft_model_state_dict
    from PIL import Image
    from torch.nn import functional as F

    train = manifest.get("train") or {}
    model_id = train.get("modelId") or "stabilityai/stable-diffusion-xl-base-1.0"
    rank = int(train.get("rank") or 32)
    alpha = int(train.get("alpha") or 16)
    lr = float(train.get("learningRate") or 1e-4)
    max_steps = int(train.get("maxSteps") or 800)
    resolution = int(train.get("resolution") or 1024)

    props = torch.cuda.get_device_properties(0)
    vram_gb = props.total_memory / (1024**3)
    if vram_gb < 12:
        resolution = min(resolution, 768)
        rank = min(rank, 16)

    print(f"loading {model_id} at {resolution}px, rank {rank}, steps {max_steps}")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
    )
    pipe.to("cuda")
    pipe.vae.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)
    pipe.text_encoder_2.requires_grad_(False)
    pipe.unet.requires_grad_(False)
    pipe.unet.add_adapter(
        LoraConfig(
            r=rank,
            lora_alpha=alpha,
            init_lora_weights="gaussian",
            target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        )
    )
    pipe.unet.enable_gradient_checkpointing()
    for name, param in pipe.unet.named_parameters():
        if "lora" in name.lower():
            param.requires_grad_(True)

    trainable = [param for param in pipe.unet.parameters() if param.requires_grad]
    if not trainable:
        return fail("LoRA parameters were not created", 2)
    optimizer = torch.optim.AdamW(trainable, lr=lr)
    noise_scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")

    captions = []
    for image_path in images:
        text_path = image_path.with_suffix(".txt")
        captions.append(text_path.read_text(encoding="utf-8").strip() if text_path.exists() else manifest["caption"])

    pipe.unet.train()
    step = 0
    while step < max_steps:
        image_path = images[step % len(images)]
        caption = captions[step % len(captions)]
        image = Image.open(image_path).convert("RGB").resize((resolution, resolution), Image.Resampling.BICUBIC)
        pixel = torch.from_numpy(np.asarray(image, dtype=np.float32)).permute(2, 0, 1).unsqueeze(0)
        pixel = pixel / 127.5 - 1.0
        pixel = pixel.to("cuda", dtype=torch.float32)

        with torch.no_grad():
            latents = pipe.vae.encode(pixel).latent_dist.sample()
            latents = latents * pipe.vae.config.scaling_factor
            latents = latents.to(dtype=torch.float16)
            prompt_embeds, _, pooled, _ = pipe.encode_prompt(
                prompt=caption,
                prompt_2=caption,
                device="cuda",
                num_images_per_prompt=1,
                do_classifier_free_guidance=False,
            )

        noise = torch.randn_like(latents)
        timesteps = torch.randint(
            0, noise_scheduler.config.num_train_timesteps, (1,), device=latents.device
        ).long()
        noisy = noise_scheduler.add_noise(latents, noise, timesteps)
        time_ids = torch.tensor(
            [[resolution, resolution, 0, 0, resolution, resolution]],
            device=latents.device,
            dtype=torch.float16,
        )
        pred = pipe.unet(
            noisy,
            timesteps,
            encoder_hidden_states=prompt_embeds.to(dtype=torch.float16),
            added_cond_kwargs={
                "text_embeds": pooled.to(dtype=torch.float16),
                "time_ids": time_ids,
            },
        ).sample
        target = noise
        if getattr(noise_scheduler.config, "prediction_type", "epsilon") == "v_prediction":
            target = noise_scheduler.get_velocity(latents, noise, timesteps)
        loss = F.mse_loss(pred.float(), target.float())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1
        if step == 1 or step % 50 == 0 or step == max_steps:
            print(f"step {step}/{max_steps} loss {loss.item():.4f}")

    adapter_dir = Path(manifest["outputDir"]) / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    unet_lora_layers = get_peft_model_state_dict(pipe.unet)
    StableDiffusionXLPipeline.save_lora_weights(str(adapter_dir), unet_lora_layers=unet_lora_layers)
    (adapter_dir / "method.json").write_text(
        json.dumps({"method": "lora-txt2img", "compositing": False, "modelId": model_id}, indent=2),
        encoding="utf-8",
    )
    print(f"saved LoRA adapter: {adapter_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
