import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from safetensors.torch import save_file
from huggingface_hub import create_repo, upload_folder
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
from diffusers.utils import convert_state_dict_to_diffusers
from PIL import Image


def resolve_lora_init_path(init_path: str | Path) -> str:
    path = Path(init_path)
    candidate_paths = []

    def add_candidate(candidate: Path) -> None:
        if candidate not in candidate_paths:
            candidate_paths.append(candidate)

    # 1) Direct adapter export directory, e.g. final_lora/ or checkpoints/.../hf_checkpoint/lora/
    add_candidate(path)

    # 2) A Lightning .ckpt file path or checkpoint directory beside the exported HF adapter.
    if path.suffix == ".ckpt":
        checkpoint_dir = path.parent
        add_candidate(checkpoint_dir / "hf_checkpoint" / "lora")
        add_candidate(checkpoint_dir / "lora")
    else:
        add_candidate(path / "hf_checkpoint" / "lora")
        add_candidate(path / "lora")

    # 3) Common project export layout under outputs/*/final_lora.
    add_candidate(path / "final_lora")

    adapter_filenames = (
        "pytorch_lora_weights.safetensors",
        "pytorch_lora_weights.bin",
        "adapter_model.safetensors",
        "adapter_model.bin",
    )

    for candidate in candidate_paths:
        if candidate.is_file():
            if candidate.name in adapter_filenames:
                return str(candidate.parent)
            continue

        if not candidate.is_dir():
            continue

        if any((candidate / filename).is_file() for filename in adapter_filenames):
            return str(candidate)

    searched_locations = "\n - ".join(str(candidate) for candidate in candidate_paths)
    raise FileNotFoundError(
        "LoRA init path not found. Expected a Diffusers LoRA adapter directory or a checkpoint-related path. "
        f"Input: {init_path}\nSearched:\n - {searched_locations}"
    )


def build_diffusers_lora_state_dicts(
        *,
        unet: Any,
        train_unet_lora: bool,
        text_encoder: Any = None,
        train_text_encoder_lora: bool = False,
) -> tuple[Optional[Dict[str, torch.Tensor]], Optional[Dict[str, torch.Tensor]]]:
    """Collect diffusers-formatted LoRA weights for pipeline.save_lora_weights."""
    unet_state_dict: Optional[Dict[str, torch.Tensor]] = None
    text_encoder_state_dict: Optional[Dict[str, torch.Tensor]] = None

    if train_unet_lora:
        unet_state_dict = {
            key: value.detach().cpu()
            for key, value in convert_state_dict_to_diffusers(unet.get_adapter_state_dict()).items()
        }

    if train_text_encoder_lora and text_encoder is not None:
        text_encoder_state_dict = {
            key: value.detach().cpu()
            for key, value in convert_state_dict_to_diffusers(text_encoder.get_adapter_state_dict()).items()
        }

    return unet_state_dict, text_encoder_state_dict


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

    validation_seed = validation_cfg.get("seed", 42)
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
                height=int(validation_cfg.get("height", 512)),
                width=int(validation_cfg.get("width", 512)),
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


def upload_model_artifacts_to_hub(
        *,
        hub_cfg: Dict[str, Any],
        export_dir: str | Path,
        rank_zero_warn_fn: Any,
        rank_zero_info_fn: Any,
) -> None:
    """Upload exported artifacts to the Hugging Face Hub when configured."""
    if not bool(hub_cfg.get("push_to_hub", False)):
        return

    token = hub_cfg.get("token") or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    repo_id = hub_cfg.get("model_id")
    if not repo_id:
        rank_zero_warn_fn("[Hub] `hub.model_id` is missing, skipping upload.")
        return

    repo_id = str(repo_id)
    create_repo(repo_id=repo_id, token=token, exist_ok=True)

    upload_folder(
        repo_id=repo_id,
        folder_path=str(export_dir),
        token=token,
        commit_message="Upload Diffusers model artifacts",
    )
    rank_zero_info_fn(f"[Hub] Uploaded artifacts to https://huggingface.co/{repo_id}")
