import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DPMSolverMultistepScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from pytorch_lightning.utilities.rank_zero import rank_zero_info
from PIL import Image
from transformers import CLIPTextModel, CLIPTokenizer


class StableDiffusionLightningModule(pl.LightningModule):
    """Lightning module for SD1.5 denoising training in latent space."""

    def __init__(self, cfg: Dict[str, Any]) -> None:
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters(cfg)

        model_name = cfg.get("pretrained_model_name_or_path")
        clip_model_name = cfg.get("clip_model_name_or_path", "CompVis/stable-diffusion-v1-4")
        vae_name = cfg.get("vae_model_name_or_path", "stabilityai/sd-vae-ft-mse")
        unet_config_path = cfg.get("unet_config_path")
        scheduler_config_path = cfg.get("scheduler_config_path")

        if model_name:
            self.tokenizer = CLIPTokenizer.from_pretrained(model_name, subfolder="tokenizer")
            self.text_encoder = CLIPTextModel.from_pretrained(model_name, subfolder="text_encoder")
            self.noise_scheduler = DDPMScheduler.from_pretrained(model_name, subfolder="scheduler")
            self.vae = AutoencoderKL.from_pretrained(model_name, subfolder="vae")
            self.unet = UNet2DConditionModel.from_pretrained(model_name, subfolder="unet")
        else:
            self.tokenizer = CLIPTokenizer.from_pretrained(clip_model_name, subfolder="tokenizer")
            self.text_encoder = CLIPTextModel.from_pretrained(clip_model_name, subfolder="text_encoder")
            self.vae = AutoencoderKL.from_pretrained(vae_name)

            if scheduler_config_path:
                self.noise_scheduler = DDPMScheduler.from_config(scheduler_config_path)
            else:
                raise ValueError("scheduler_config_path must be provided when pretrained_model_name_or_path is empty")

            if unet_config_path:
                with open(unet_config_path, "r", encoding="utf-8") as f:
                    raw_unet_config = json.load(f)
                unet_config = {k: v for k, v in raw_unet_config.items() if not k.startswith("_")}
                self.unet = UNet2DConditionModel.from_config(unet_config)
            else:
                raise ValueError("unet_config_path must be provided when pretrained_model_name_or_path is empty")

        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.unet.requires_grad_(True)

        self.training_cfg = cfg["training"]
        self.validation_cfg = cfg.get("validation", {})
        self.use_spatial_conditioning = bool(self.training_cfg.get("use_spatial_conditioning", True))
        self.use_aspect_ratio_conditioning = bool(self.training_cfg.get("use_aspect_ratio_conditioning", True))
        self.vae_scale_factor = int(2 ** (len(self.vae.config.block_out_channels) - 1))

        self.aspect_ratio_mlp: Optional[nn.Module]
        if self.use_aspect_ratio_conditioning:
            self.aspect_ratio_mlp = nn.Sequential(
                nn.Linear(1, 32),
                nn.SiLU(),
                nn.Linear(32, 1),
            )
        else:
            self.aspect_ratio_mlp = None

        if self.use_spatial_conditioning or self.use_aspect_ratio_conditioning:
            self._expand_unet_input_channels(
                int(self.unet.config.in_channels)
                + (2 if self.use_spatial_conditioning else 0)
                + (1 if self.use_aspect_ratio_conditioning else 0)
            )

        # Runtime metric cache for callback-side unified stdout logging.
        self.runtime_log_dict: Dict[str, float] = {}
        self.loss_ema: Optional[float] = None

        if bool(self.training_cfg.get("gradient_checkpointing", False)):
            self.unet.enable_gradient_checkpointing()

        if bool(self.training_cfg.get("enable_xformers_memory_efficient_attention", False)):
            try:
                self.unet.enable_xformers_memory_efficient_attention()
            except Exception as exc:
                rank_zero_info(
                    f"[warning] xFormers attention requested but unavailable; falling back to standard attention. ({exc})"
                )

    def on_fit_start(self) -> None:
        # Keep frozen modules in eval mode during training.
        self.vae.eval()
        self.text_encoder.eval()

    @torch.no_grad()
    def encode_prompt(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.text_encoder(input_ids, attention_mask=None)[0]

    def _compute_snr(self, timesteps: torch.Tensor) -> torch.Tensor:
        alphas_cumprod = self.noise_scheduler.alphas_cumprod.to(device=timesteps.device, dtype=torch.float32)
        alpha = alphas_cumprod[timesteps]
        sigma = (1.0 - alpha).clamp(min=1e-8)
        return alpha / sigma

    def _expand_unet_input_channels(self, target_in_channels: int) -> None:
        current_in_channels = int(self.unet.config.in_channels)
        if target_in_channels <= current_in_channels:
            return

        conv_in = self.unet.conv_in
        expanded_conv = nn.Conv2d(
            in_channels=target_in_channels,
            out_channels=conv_in.out_channels,
            kernel_size=conv_in.kernel_size,
            stride=conv_in.stride,
            padding=conv_in.padding,
        )
        with torch.no_grad():
            expanded_conv.weight.zero_()
            expanded_conv.bias.copy_(conv_in.bias)
            expanded_conv.weight[:, :current_in_channels].copy_(conv_in.weight)

        self.unet.conv_in = expanded_conv
        self.unet.config.in_channels = target_in_channels
        if hasattr(self.unet, "register_to_config"):
            self.unet.register_to_config(in_channels=target_in_channels)

    def _build_pos_map(self, batch: Dict[str, torch.Tensor], latent_height: int, latent_width: int) -> Optional[torch.Tensor]:
        if not self.use_spatial_conditioning:
            return None

        original_size = batch.get("original_size")
        crop_top_left = batch.get("crop_top_left")
        target_size = batch.get("target_size")
        if original_size is None or crop_top_left is None or target_size is None:
            return None

        pos_maps: List[torch.Tensor] = []
        for crop, target in zip(crop_top_left, target_size):
            target_h = max(int(target[0].item()), 1)
            target_w = max(int(target[1].item()), 1)
            latent_target_h = max(int(round(target_h / self.vae_scale_factor)), latent_height)
            latent_target_w = max(int(round(target_w / self.vae_scale_factor)), latent_width)
            crop_top = max(0, min(int(round(crop[0].item() / self.vae_scale_factor)), max(0, latent_target_h - latent_height)))
            crop_left = max(0, min(int(round(crop[1].item() / self.vae_scale_factor)), max(0, latent_target_w - latent_width)))

            y_coords = torch.linspace(-1.0, 1.0, steps=latent_target_h, device=self.device, dtype=torch.float32)
            x_coords = torch.linspace(-1.0, 1.0, steps=latent_target_w, device=self.device, dtype=torch.float32)
            grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing="ij")
            pos_map = torch.stack((grid_y, grid_x), dim=0)
            pos_map = pos_map[:, crop_top: crop_top + latent_height, crop_left: crop_left + latent_width]
            if pos_map.shape[-2:] != (latent_height, latent_width):
                pos_map = F.interpolate(pos_map.unsqueeze(0), size=(latent_height, latent_width), mode="bilinear", align_corners=False).squeeze(0)
            pos_maps.append(pos_map)

        return torch.stack(pos_maps, dim=0)

    def _build_aspect_ratio_map(self, batch: Dict[str, torch.Tensor], latent_height: int, latent_width: int) -> Optional[torch.Tensor]:
        if not self.use_aspect_ratio_conditioning:
            return None

        aspect_ratio = batch.get("aspect_ratio")
        if aspect_ratio is None:
            return None

        aspect_ratio = aspect_ratio.to(device=self.device).view(-1, 1)
        if self.aspect_ratio_mlp is not None:
            mlp_param = next(self.aspect_ratio_mlp.parameters())
            aspect_ratio = aspect_ratio.to(device=mlp_param.device, dtype=mlp_param.dtype)
            aspect_ratio = self.aspect_ratio_mlp(aspect_ratio)
        return aspect_ratio.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, latent_height, latent_width)

    def _augment_unet_input(self, noisy_latents: torch.Tensor, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        conditioning_tensors: List[torch.Tensor] = [noisy_latents]
        latent_height, latent_width = noisy_latents.shape[-2:]

        pos_map = self._build_pos_map(batch, latent_height, latent_width)
        if pos_map is not None:
            conditioning_tensors.append(pos_map.to(device=noisy_latents.device, dtype=noisy_latents.dtype))

        aspect_ratio_map = self._build_aspect_ratio_map(batch, latent_height, latent_width)
        if aspect_ratio_map is not None:
            conditioning_tensors.append(aspect_ratio_map.to(device=noisy_latents.device, dtype=noisy_latents.dtype))

        if len(conditioning_tensors) == 1:
            return noisy_latents
        return torch.cat(conditioning_tensors, dim=1)

    def _build_generation_conditioning(
        self,
        batch_size: int,
        latent_height: int,
        latent_width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        conditioning_tensors: List[torch.Tensor] = []

        if self.use_spatial_conditioning:
            conditioning_tensors.append(
                torch.zeros(batch_size, 2, latent_height, latent_width, device=device, dtype=dtype)
            )

        if self.use_aspect_ratio_conditioning:
            aspect_ratio = torch.ones(batch_size, 1, device=device, dtype=dtype)
            if self.aspect_ratio_mlp is not None:
                mlp_param = next(self.aspect_ratio_mlp.parameters())
                aspect_ratio = aspect_ratio.to(device=mlp_param.device, dtype=mlp_param.dtype)
                aspect_ratio = self.aspect_ratio_mlp(aspect_ratio)
                aspect_ratio = aspect_ratio.to(device=device, dtype=dtype)
            conditioning_tensors.append(
                aspect_ratio.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, latent_height, latent_width)
            )

        if len(conditioning_tensors) == 0:
            return None
        return torch.cat(conditioning_tensors, dim=1)

    def _compute_loss(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, Any]]:
        pixel_values = batch["pixel_values"]
        input_ids = batch["input_ids"]

        # VAE expects [B, C, H, W]. If an upstream batch of size 1 was flattened
        # to [C, H, W], recover the missing batch dimension.
        if pixel_values.ndim == 3:
            pixel_values = pixel_values.unsqueeze(0)
        elif pixel_values.ndim != 4:
            raise ValueError(f"Expected pixel_values to be 4D [B, C, H, W], got shape {tuple(pixel_values.shape)}")

        if input_ids.ndim == 3:
            input_ids = input_ids.squeeze(1)
        elif input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)

        latents = self.vae.encode(pixel_values).latent_dist.sample()
        latents = latents * self.vae.config.scaling_factor

        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (bsz,),
            device=latents.device,
            dtype=torch.long,
        )

        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
        encoder_hidden_states = self.encode_prompt(input_ids)
        unet_input = self._augment_unet_input(noisy_latents, batch)

        model_pred = self.unet(
            unet_input,
            timesteps,
            encoder_hidden_states,
        ).sample

        prediction_type = self.noise_scheduler.config.prediction_type
        if prediction_type == "epsilon":
            target = noise
        elif prediction_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type: {prediction_type}")

        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
        snr = self._compute_snr(timesteps)

        metrics: Dict[str, Any] = {
            "train/loss": loss.detach(),
            "train/batch_size": float(bsz),
            "train/timestep_mean": timesteps.detach().float().mean(),
            "train/timestep_std": timesteps.detach().float().std(unbiased=False),
            "train/snr_mean": snr.detach().mean(),
            "train/latent_std": latents.detach().float().std(unbiased=False),
            "train/noisy_latent_std": noisy_latents.detach().float().std(unbiased=False),
            "train/unet_input_std": unet_input.detach().float().std(unbiased=False),
            "train/noise_std": noise.detach().float().std(unbiased=False),
            "train/pred_std": model_pred.detach().float().std(unbiased=False),
            "train/target_std": target.detach().float().std(unbiased=False),
            "train/pred_target_mse": (model_pred.detach().float() - target.detach().float()).pow(2).mean(),
        }
        return loss, metrics

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, Any]:
        loss, train_metrics = self._compute_loss(batch)

        loss_value = float(loss.detach().float().cpu().item())
        ema_decay = float(self.training_cfg.get("loss_ema_decay", 0.95))
        if self.loss_ema is None:
            self.loss_ema = loss_value
        else:
            self.loss_ema = ema_decay * self.loss_ema + (1.0 - ema_decay) * loss_value
        train_metrics["train/loss_ema"] = self.loss_ema

        if self.trainer.optimizers:
            train_metrics["train/lr"] = float(self.trainer.optimizers[0].param_groups[0]["lr"])

        self.runtime_log_dict = {
            key: float(value.detach().float().cpu().item()) if torch.is_tensor(value) else float(value)
            for key, value in train_metrics.items()
        }

        return {
            "loss": loss,
            "train_metrics": train_metrics,
        }

    @torch.no_grad()
    def on_train_batch_end(self, outputs: Any, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        if not self.validation_cfg.get("enabled", True):
            return

        if self.trainer is not None and not self.trainer.is_global_zero:
            return

        every_n_steps = int(self.validation_cfg.get("every_n_steps", 0))
        if every_n_steps <= 0:
            return

        global_step = int(self.global_step)
        if global_step == 0 or global_step % every_n_steps != 0:
            return

        prompts = self.validation_cfg.get("prompts", [])
        if len(prompts) == 0:
            return

        pipeline = StableDiffusionPipeline(
            vae=self.vae,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            unet=self.unet,
            scheduler=DPMSolverMultistepScheduler.from_config(self.noise_scheduler.config),
            safety_checker=None,
            feature_extractor=None,
            requires_safety_checker=False,
        )
        pipeline = pipeline.to(self.device)
        pipeline.set_progress_bar_config(disable=True)

        original_forward = pipeline.unet.forward
        original_in_channels = int(pipeline.unet.config.in_channels)

        if self.use_spatial_conditioning or self.use_aspect_ratio_conditioning:
            pipeline.unet.config.in_channels = 4
            if hasattr(pipeline.unet, "register_to_config"):
                pipeline.unet.register_to_config(in_channels=4)

            def conditioned_forward(sample: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
                conditioning = self._build_generation_conditioning(
                    batch_size=sample.shape[0],
                    latent_height=sample.shape[-2],
                    latent_width=sample.shape[-1],
                    device=sample.device,
                    dtype=sample.dtype,
                )
                if conditioning is not None:
                    sample = torch.cat((sample, conditioning), dim=1)
                return original_forward(sample, *args, **kwargs)

            pipeline.unet.forward = conditioned_forward

        images: List[Image.Image] = []
        try:
            for prompt in prompts:
                image = pipeline(
                    prompt,
                    num_inference_steps=int(self.validation_cfg.get("num_inference_steps", 25)),
                    guidance_scale=float(self.validation_cfg.get("guidance_scale", 7.5)),
                ).images[0]
                images.append(image)
        finally:
            pipeline.unet.forward = original_forward
            pipeline.unet.config.in_channels = original_in_channels
            if hasattr(pipeline.unet, "register_to_config"):
                pipeline.unet.register_to_config(in_channels=original_in_channels)

        # Lightning TensorBoard logger exposes underlying SummaryWriter.
        if self.logger is not None and hasattr(self.logger, "experiment"):
            import numpy as np

            writer = self.logger.experiment
            for idx, image in enumerate(images):
                image_tensor = torch.from_numpy(np.asarray(image)).permute(2, 0, 1).float() / 255.0
                writer.add_image(f"validation/sample_{idx}", image_tensor, global_step)

        del pipeline

    def save_pretrained(self, save_directory: str) -> None:
        """Export current training state to a Diffusers-compatible checkpoint folder."""
        save_dir = Path(save_directory)
        save_dir.mkdir(parents=True, exist_ok=True)

        pipeline = StableDiffusionPipeline(
            vae=self.vae,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            unet=self.unet,
            scheduler=DPMSolverMultistepScheduler.from_config(self.noise_scheduler.config),
            safety_checker=None,
            feature_extractor=None,
            requires_safety_checker=False,
        )
        pipeline.save_pretrained(str(save_dir), safe_serialization=True)

        with open(save_dir / "training_config.json", "w", encoding="utf-8") as f:
            json.dump(self.cfg, f, ensure_ascii=False, indent=2)

    def save_hf_checkpoint(self, checkpoint_filepath: str) -> None:
        """Export UNet-style HF checkpoint colocated with a Lightning .ckpt filepath."""
        checkpoint_dir = Path(checkpoint_filepath).with_suffix("")
        hf_dir = checkpoint_dir / "hf_checkpoint"
        hf_dir.mkdir(parents=True, exist_ok=True)

        unet_save_dir = hf_dir / "unet"
        self.unet.save_pretrained(str(unet_save_dir))

        use_ema = bool(getattr(self, "use_ema", False))
        if use_ema:
            ema_obj = getattr(self, "ema", None)
            ema_unet = getattr(self, "ema_unet", None)
            ema_save_dir = hf_dir / "unet_ema"

            if ema_obj is not None and hasattr(ema_obj, "save_pretrained"):
                ema_obj.save_pretrained(str(ema_save_dir))
            elif ema_unet is not None and hasattr(ema_unet, "save_pretrained"):
                ema_unet.save_pretrained(str(ema_save_dir))

        with open(hf_dir / "training_config.json", "w", encoding="utf-8") as f:
            json.dump(self.cfg, f, indent=2, ensure_ascii=False)

    def configure_optimizers(self):
        use_8bit_adam = bool(self.training_cfg.get("use_8bit_adam", False))
        optimizer_cls = torch.optim.AdamW

        if use_8bit_adam:
            try:
                import bitsandbytes as bnb
            except ImportError:
                raise ImportError(
                    "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
                )
            optimizer_cls = bnb.optim.AdamW8bit

        optimizer = optimizer_cls(
            self.unet.parameters(),
            lr=float(self.training_cfg.get("learning_rate", 1e-4)),
            betas=(
                float(self.training_cfg.get("adam_beta1", 0.9)),
                float(self.training_cfg.get("adam_beta2", 0.999)),
            ),
            eps=float(self.training_cfg.get("adam_epsilon", 1e-8)),
            weight_decay=float(self.training_cfg.get("weight_decay", 0.01)),
        )

        estimated_steps = self.trainer.estimated_stepping_batches
        warmup_steps = int(self.training_cfg.get("lr_warmup_steps", 0))
        lr_scheduler_type = self.training_cfg.get("lr_scheduler", "cosine")

        lr_scheduler = get_scheduler(
            lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=max(1, estimated_steps),
        )

        scheduler_cfg = {
            "scheduler": lr_scheduler,
            "interval": "step",
            "frequency": 1,
        }

        return {"optimizer": optimizer, "lr_scheduler": scheduler_cfg}
