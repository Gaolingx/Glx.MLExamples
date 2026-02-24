import json
from typing import Any, Dict, List

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DPMSolverMultistepScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from PIL import Image
from transformers import CLIPTextModel, CLIPTokenizer


class StableDiffusionLightningModule(pl.LightningModule):
    """Lightning module for SD1.5 denoising training in latent space."""

    def __init__(self, cfg: Dict[str, Any]) -> None:
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters(cfg)

        model_name = cfg["pretrained_model_name_or_path"]
        clip_model_name = cfg.get("clip_model_name_or_path", "CompVis/stable-diffusion-v1-4")
        vae_name = cfg.get("vae_model_name_or_path", "stabilityai/sd-vae-ft-mse")

        self.tokenizer = CLIPTokenizer.from_pretrained(clip_model_name, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(clip_model_name, subfolder="text_encoder")
        self.noise_scheduler = DDPMScheduler.from_pretrained(model_name, subfolder="scheduler")
        self.vae = AutoencoderKL.from_pretrained(vae_name)

        unet_config_path = cfg.get("unet_config_path")
        if unet_config_path:
            with open(unet_config_path, "r", encoding="utf-8") as f:
                raw_unet_config = json.load(f)
            unet_config = {k: v for k, v in raw_unet_config.items() if not k.startswith("_")}
            self.unet = UNet2DConditionModel.from_config(unet_config)
        else:
            self.unet = UNet2DConditionModel.from_pretrained(model_name, subfolder="unet")

        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.unet.requires_grad_(True)

        self.training_cfg = cfg["training"]
        self.validation_cfg = cfg.get("validation", {})

        # Runtime metric cache for callback-side unified stdout logging.
        self.runtime_log_dict: Dict[str, float] = {}

        self.automatic_optimization = not bool(self.training_cfg.get("manual_optimization", False))

        if bool(self.training_cfg.get("gradient_checkpointing", False)):
            self.unet.enable_gradient_checkpointing()

        if bool(self.training_cfg.get("enable_xformers_memory_efficient_attention", False)):
            try:
                self.unet.enable_xformers_memory_efficient_attention()
            except Exception as exc:
                print(
                    f"[warning] xFormers attention requested but unavailable; falling back to standard attention. ({exc})"
                )

    def on_fit_start(self) -> None:
        # Keep frozen modules in eval mode during training.
        self.vae.eval()
        self.text_encoder.eval()

    @torch.no_grad()
    def encode_prompt(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.text_encoder(input_ids, attention_mask=None)[0]

    def _compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
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

        model_pred = self.unet(
            noisy_latents,
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

        return F.mse_loss(model_pred.float(), target.float(), reduction="mean")

    @staticmethod
    def _to_float_for_runtime_log(value: Any) -> float:
        if torch.is_tensor(value):
            value = value.detach()
            if value.numel() == 1:
                return float(value.cpu())
            return float(value.float().mean().cpu())
        return float(value)

    def _log_metric(self, name: str, value: Any, **kwargs: Any) -> None:
        self.log(name, value, **kwargs)
        self.runtime_log_dict[name] = self._to_float_for_runtime_log(value)

    def _log_train_metrics(self, loss: torch.Tensor, lr: float) -> None:
        self._log_metric("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self._log_metric("train/lr", lr, on_step=True, on_epoch=False, prog_bar=False, sync_dist=True)

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        if self.automatic_optimization:
            loss = self._compute_loss(batch)
            lr = self.trainer.optimizers[0].param_groups[0]["lr"]
            self._log_train_metrics(loss, lr)
            return loss

        optimizer = self.optimizers()
        scheduler = self.lr_schedulers()
        loss = self._compute_loss(batch)

        optimizer.zero_grad(set_to_none=True)
        self.manual_backward(loss)
        self.clip_gradients(
            optimizer,
            gradient_clip_val=float(self.training_cfg.get("max_grad_norm", 1.0)),
            gradient_clip_algorithm="norm",
        )
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        self._log_train_metrics(loss, lr)
        return loss.detach()

    @torch.no_grad()
    def on_train_batch_end(self, outputs: Any, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        if not self.validation_cfg.get("enabled", True):
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

        images: List[Image.Image] = []
        for prompt in prompts:
            image = pipeline(
                prompt,
                num_inference_steps=int(self.validation_cfg.get("num_inference_steps", 25)),
                guidance_scale=float(self.validation_cfg.get("guidance_scale", 7.5)),
            ).images[0]
            images.append(image)

        # Lightning TensorBoard logger exposes underlying SummaryWriter.
        if self.logger is not None and hasattr(self.logger, "experiment"):
            import numpy as np

            writer = self.logger.experiment
            for idx, image in enumerate(images):
                image_tensor = torch.from_numpy(np.asarray(image)).permute(2, 0, 1).float() / 255.0
                writer.add_image(f"validation/sample_{idx}", image_tensor, global_step)

        del pipeline

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
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
