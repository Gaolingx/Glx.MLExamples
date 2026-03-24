import argparse
from pathlib import Path
import sys

import torch
from diffusers import StableDiffusionImg2ImgPipeline, StableDiffusionPipeline
from PIL import Image

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.lightningmodule.sd15_module import StableDiffusionLightningModule
from src.utils.config import load_json_config
from src.utils.train_utils import build_inference_scheduler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inference for SD1.5 Lightning checkpoint.")
    parser.add_argument("--config", type=str, default="./configs/train_config.json", help="Path to JSON config file.")
    parser.add_argument("--ckpt_path", type=str, default=None, help="Lightning .ckpt path.")
    parser.add_argument("--base_model", type=str, default=None, help="Base SD model for standalone LoRA inference.")
    parser.add_argument("--lora_path", type=str, default=None, help="Optional LoRA adapter directory to load.")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt.")
    parser.add_argument("--negative_prompt", type=str, default=None, help="Negative prompt.")
    parser.add_argument("--num_images", type=int, default=1, help="Number of images.")
    parser.add_argument("--clip_skip", type=int, default=None, help="CLIP skip value. 1 means final layer, 2 means skip last layer.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for generation.")
    parser.add_argument("--sampler", type=str, default=None, help="Sampler name, e.g. dpmpp_2m, euler, euler_a, ddim.")
    parser.add_argument("--scheduler", type=str, default=None, help="Scheduler flavor, e.g. default or karras.")
    parser.add_argument("--steps", type=int, default=None, help="Inference steps.")
    parser.add_argument("--cfg_scale", type=float, default=None, help="CFG/guidance scale.")
    parser.add_argument("--height", type=int, default=None, help="Output image height.")
    parser.add_argument("--width", type=int, default=None, help="Output image width.")
    parser.add_argument("--output", type=str, default=None, help="Output image directory.")
    parser.add_argument("--init_image", type=str, default=None, help="Optional init image path for img2img.")
    parser.add_argument("--denoise_strength", type=float, default=None, help="Img2img denoise strength in [0, 1].")
    return parser.parse_args()


def load_inference_runtime_config(args: argparse.Namespace) -> dict:
    cfg = load_json_config(args.config)

    overrides = {
        "inference": {
            "negative_prompt": args.negative_prompt,
            "seed": args.seed,
            "sampler": args.sampler,
            "scheduler": args.scheduler,
            "num_inference_steps": args.steps,
            "guidance_scale": args.cfg_scale,
            "height": args.height,
            "width": args.width,
            "output_dir": args.output,
            "denoise_strength": args.denoise_strength,
        },
        "clip": {
            "skip": args.clip_skip,
        },
        "runtime": {
            "ckpt_path": args.ckpt_path,
            "base_model": args.base_model,
            "lora_path": args.lora_path,
            "init_image": args.init_image,
        },
    }

    for section, values in overrides.items():
        section_cfg = cfg.setdefault(section, {})
        for key, value in values.items():
            if value is not None:
                section_cfg[key] = value

    return cfg


@torch.no_grad()
def main() -> None:
    args = parse_args()
    cfg = load_inference_runtime_config(args)
    infer_cfg = cfg.get("inference", {})
    lora_cfg = cfg.get("lora", {})
    clip_cfg = cfg.get("clip", {})

    runtime_cfg = cfg.get("runtime", {})
    ckpt_path = runtime_cfg.get("ckpt_path")
    base_model = runtime_cfg.get("base_model")
    lora_path = runtime_cfg.get("lora_path")
    init_image_path = runtime_cfg.get("init_image")

    if ckpt_path is None and (base_model is None or lora_path is None):
        raise ValueError("Provide either --ckpt_path, or both --base_model and --lora_path.")

    if ckpt_path is not None:
        module = StableDiffusionLightningModule.load_from_checkpoint(ckpt_path, cfg=cfg, map_location="cpu")
        module.eval()

        scheduler = build_inference_scheduler(
            module.noise_scheduler.config,
            infer_cfg.get("sampler", "dpmpp_2m"),
            infer_cfg.get("scheduler", "default"),
        )
        pipe_cls = StableDiffusionImg2ImgPipeline if init_image_path else StableDiffusionPipeline
        pipe = pipe_cls(
            vae=module.vae,
            text_encoder=module.text_encoder,
            tokenizer=module.tokenizer,
            unet=module.unet,
            scheduler=scheduler,
            safety_checker=None,
            feature_extractor=None,
            requires_safety_checker=False,
        )
    else:
        pipe_cls = StableDiffusionImg2ImgPipeline if init_image_path else StableDiffusionPipeline
        pipe = pipe_cls.from_pretrained(
            base_model,
            safety_checker=None,
            feature_extractor=None,
            requires_safety_checker=False,
        )
        pipe.scheduler = build_inference_scheduler(
            pipe.scheduler.config,
            infer_cfg.get("sampler", "dpmpp_2m"),
            infer_cfg.get("scheduler", "default"),
        )
        pipe.unet.load_lora_adapter(lora_path, prefix=None, adapter_name=lora_cfg.get("adapter_name", "default"))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)

    negative_prompt = infer_cfg.get("negative_prompt")
    clip_skip = int(clip_cfg.get("skip", 1))
    seed = infer_cfg.get("seed")
    num_inference_steps = int(infer_cfg.get("num_inference_steps", 30))
    guidance_scale = float(infer_cfg.get("guidance_scale", 7.5))
    height = int(infer_cfg.get("height", 512))
    width = int(infer_cfg.get("width", 512))
    denoise_strength = float(infer_cfg.get("denoise_strength", 0.75))
    if not 0.0 <= denoise_strength <= 1.0:
        raise ValueError("denoise_strength must be in [0, 1].")

    out_dir = Path(infer_cfg.get("output_dir", "outputs/sd15-lightning/inference"))
    out_dir.mkdir(parents=True, exist_ok=True)

    init_image = None
    if init_image_path is not None:
        init_image = Image.open(init_image_path).convert("RGB")
        if "height" not in infer_cfg or "width" not in infer_cfg:
            width, height = init_image.size

    generator_device = device

    for idx in range(args.num_images):
        image_seed = None if seed is None else int(seed) + idx
        generator = None
        if image_seed is not None:
            generator = torch.Generator(device=generator_device).manual_seed(image_seed)

        pipeline_kwargs = {
            "prompt": args.prompt,
            "negative_prompt": negative_prompt,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "clip_skip": clip_skip,
            "generator": generator,
        }
        if init_image is not None:
            pipeline_kwargs["image"] = init_image.resize((width, height), Image.LANCZOS)
            pipeline_kwargs["strength"] = denoise_strength
        else:
            pipeline_kwargs["height"] = height
            pipeline_kwargs["width"] = width

        image = pipe(**pipeline_kwargs).images[0]
        save_path = out_dir / f"sample_{idx:03d}.png"
        image.save(save_path)


if __name__ == "__main__":
    main()
