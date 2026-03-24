from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from diffusers import (
    DDPMScheduler,
    DDIMScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    HeunDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
    DPMSolverMultistepScheduler,
    StableDiffusionPipeline,
)
from PIL import Image


def resolve_lora_init_path(init_path: str | Path) -> str:
    path = Path(init_path)

    if path.is_dir():
        candidate_dirs = [
            path,
            path / "unet_lora",
            path / "hf_checkpoint" / "unet_lora",
        ]
        for candidate in candidate_dirs:
            if candidate.is_dir():
                return str(candidate)

    if path.is_file() and path.suffix == ".ckpt":
        checkpoint_dir = path.with_suffix("")
        candidate_dirs = [
            checkpoint_dir / "hf_checkpoint" / "unet_lora",
            path.parent / "hf_checkpoint" / "unet_lora",
        ]
        for candidate in candidate_dirs:
            if candidate.is_dir():
                return str(candidate)
        raise FileNotFoundError(
            "LoRA init checkpoint was provided, but no exported adapter directory was found next to it. "
            "Expected `hf_checkpoint/unet_lora`."
        )

    if path.exists():
        return str(path)

    raise FileNotFoundError(f"LoRA init path not found: {init_path}")


def build_inference_scheduler(base_config, sampler_name: Optional[str], scheduler_name: Optional[str]):
    sampler_key = str(sampler_name or "dpmpp_2m").strip().lower()
    scheduler_key = str(scheduler_name or "default").strip().lower()

    scheduler_map = {
        "ddim": DDIMScheduler,
        "ddpm": DDPMScheduler,
        "dpm": DPMSolverMultistepScheduler,
        "dpmpp_2m": DPMSolverMultistepScheduler,
        "euler": EulerDiscreteScheduler,
        "euler_a": EulerAncestralDiscreteScheduler,
        "heun": HeunDiscreteScheduler,
        "lms": LMSDiscreteScheduler,
        "pndm": PNDMScheduler,
    }

    scheduler_cls = scheduler_map.get(sampler_key)
    if scheduler_cls is None:
        raise ValueError(f"Unsupported sampler: {sampler_name}")

    if scheduler_cls is DPMSolverMultistepScheduler:
        if scheduler_key not in {"default", "karras", "sgm_uniform", "exponential"}:
            raise ValueError(f"Unsupported scheduler type for DPM sampler: {scheduler_name}")

        update_kwargs = {"algorithm_type": "dpmsolver++"}
        if scheduler_key == "karras":
            update_kwargs["use_karras_sigmas"] = True
        elif scheduler_key == "sgm_uniform":
            update_kwargs["timestep_spacing"] = "trailing"
        elif scheduler_key == "exponential":
            update_kwargs["use_exponential_sigmas"] = True

        return DPMSolverMultistepScheduler.from_config(base_config, **update_kwargs)

    return scheduler_cls.from_config(base_config)


def run_validation_inference(
        *,
        vae: Any,
        text_encoder: Any,
        tokenizer: Any,
        unet: Any,
        device: torch.device,
        clip_skip: int,
        validation_cfg: Dict[str, Any],
        training_cfg: Dict[str, Any],
        inference_cfg: Dict[str, Any],
        scheduler: Any,
) -> List[Image.Image]:
    """Run validation-time inference and return generated images."""
    prompts = validation_cfg.get("prompts", [])
    if len(prompts) == 0:
        return []

    pipeline = StableDiffusionPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=scheduler,
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=False,
    )
    pipeline = pipeline.to(device)
    pipeline.set_progress_bar_config(disable=True)

    validation_seed = validation_cfg.get("seed", training_cfg.get("seed", 42))
    generator_device = device.type if isinstance(device, torch.device) else str(device)

    images: List[Image.Image] = []
    try:
        for idx, prompt in enumerate(prompts):
            prompt_seed = None if validation_seed is None else int(validation_seed) + idx
            generator = None
            if prompt_seed is not None:
                generator = torch.Generator(device=generator_device).manual_seed(prompt_seed)

            image = pipeline(
                prompt,
                num_inference_steps=int(validation_cfg.get("num_inference_steps", 25)),
                guidance_scale=float(validation_cfg.get("guidance_scale", 7.5)),
                negative_prompt=validation_cfg.get("negative_prompt"),
                height=int(validation_cfg.get("height", inference_cfg.get("height", 512))),
                width=int(validation_cfg.get("width", inference_cfg.get("width", 512))),
                clip_skip=clip_skip,
                generator=generator,
            ).images[0]
            images.append(image)
    finally:
        del pipeline

    return images


def log_validation_images(logger: Optional[Any], images: List[Image.Image], global_step: int) -> None:
    """Write generated validation images to the active TensorBoard logger."""
    if logger is None or not hasattr(logger, "experiment") or len(images) == 0:
        return

    writer = logger.experiment
    for idx, image in enumerate(images):
        image_tensor = torch.from_numpy(np.asarray(image)).permute(2, 0, 1).float() / 255.0
        writer.add_image(f"validation/sample_{idx}", image_tensor, global_step)
