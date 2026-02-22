import argparse
from pathlib import Path
import sys

import torch
from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.lightningmodule.sd15_module import StableDiffusionLightningModule
from src.utils.config import load_json_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inference for SD1.5 Lightning checkpoint.")
    parser.add_argument("--config", type=str, default="./configs/train_config.json", help="Path to JSON config file.")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Lightning .ckpt path.")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt.")
    parser.add_argument("--negative_prompt", type=str, default=None, help="Negative prompt.")
    parser.add_argument("--num_images", type=int, default=1, help="Number of images.")
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    cfg = load_json_config(args.config)

    module = StableDiffusionLightningModule.load_from_checkpoint(args.ckpt_path, cfg=cfg, map_location="cpu")
    module.eval()

    scheduler = DPMSolverMultistepScheduler.from_config(module.noise_scheduler.config)
    pipe = StableDiffusionPipeline(
        vae=module.vae,
        text_encoder=module.text_encoder,
        tokenizer=module.tokenizer,
        unet=module.unet,
        scheduler=scheduler,
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=False,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)

    infer_cfg = cfg.get("inference", {})
    negative_prompt = args.negative_prompt
    if negative_prompt is None:
        negative_prompt = infer_cfg.get("negative_prompt")

    out_dir = Path(infer_cfg.get("output_dir", "outputs/sd15-lightning/inference"))
    out_dir.mkdir(parents=True, exist_ok=True)

    for idx in range(args.num_images):
        image = pipe(
            prompt=args.prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=int(infer_cfg.get("num_inference_steps", 30)),
            guidance_scale=float(infer_cfg.get("guidance_scale", 7.5)),
            height=int(infer_cfg.get("height", 512)),
            width=int(infer_cfg.get("width", 512)),
        ).images[0]
        save_path = out_dir / f"sample_{idx:03d}.png"
        image.save(save_path)


if __name__ == "__main__":
    main()
