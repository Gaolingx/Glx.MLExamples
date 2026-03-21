import json
import importlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
from pytorch_lightning.utilities.rank_zero import rank_zero_info
from PIL import Image
from transformers import CLIPTextModel, CLIPTokenizer

from src.utils.config import load_json_config, save_json_config


class StableDiffusionLightningModule(pl.LightningModule):
    """Lightning module for SD1.5 denoising training in latent space."""

    def __init__(self, cfg: Dict[str, Any]) -> None:
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters(cfg)
        self.lora_cfg = cfg.get("lora", {})
        self.lora_enabled = bool(self.lora_cfg.get("enabled", False))
        self.lora_adapter_name = str(self.lora_cfg.get("adapter_name", "default"))

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
                 scheduler_config = load_json_config(scheduler_config_path)
                 self.noise_scheduler = DDPMScheduler.from_config(scheduler_config)
            else:
                raise ValueError("scheduler_config_path must be provided when pretrained_model_name_or_path is empty")

            if unet_config_path:
                raw_unet_config = load_json_config(unet_config_path)
                unet_config = {k: v for k, v in raw_unet_config.items() if not k.startswith("_")}
                self.unet = UNet2DConditionModel.from_config(unet_config)
            else:
                raise ValueError("unet_config_path must be provided when pretrained_model_name_or_path is empty")

        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        if self.lora_enabled:
            self._setup_lora()
        else:
            self.unet.requires_grad_(True)

        self.training_cfg = cfg["training"]
        self.validation_cfg = cfg.get("validation", {})

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

    def _setup_lora(self) -> None:
        try:
            LoraConfig = importlib.import_module("peft").LoraConfig
        except ImportError as exc:
            raise ImportError("LoRA training requires `peft`. Please install it with `pip install peft`.") from exc

        if not self.cfg.get("pretrained_model_name_or_path"):
            raise ValueError("`pretrained_model_name_or_path` must be provided when `lora.enabled=true`.")

        init_path = self.lora_cfg.get("init_path")
        resolved_init_path = self._resolve_lora_init_path(init_path) if init_path else None

        if resolved_init_path is None:
            target_modules = self.lora_cfg.get("target_modules", ["to_q", "to_k", "to_v", "to_out.0"])
            lora_config = LoraConfig(
                r=int(self.lora_cfg.get("rank", 16)),
                lora_alpha=int(self.lora_cfg.get("alpha", 16)),
                lora_dropout=float(self.lora_cfg.get("dropout", 0.0)),
                bias=str(self.lora_cfg.get("bias", "none")),
                target_modules=target_modules,
                init_lora_weights=self.lora_cfg.get("init_lora_weights", True),
            )
            self.unet.add_adapter(lora_config, adapter_name=self.lora_adapter_name)
            rank_zero_info(
                f"[LoRA] Added new adapter '{self.lora_adapter_name}' with rank={lora_config.r}, alpha={lora_config.lora_alpha}."
            )
        else:
            self.unet.load_lora_adapter(
                resolved_init_path,
                prefix=None,
                adapter_name=self.lora_adapter_name,
            )
            rank_zero_info(f"[LoRA] Loaded adapter '{self.lora_adapter_name}' from {resolved_init_path}.")

        self.unet.set_adapter(self.lora_adapter_name)
        self.unet.requires_grad_(False)
        for name, parameter in self.unet.named_parameters():
            parameter.requires_grad = "lora_" in name

        trainable_params = sum(param.numel() for param in self.unet.parameters() if param.requires_grad)
        all_params = sum(param.numel() for param in self.unet.parameters())
        rank_zero_info(
            f"[LoRA] Training {trainable_params:,} / {all_params:,} UNet parameters "
            f"({(100.0 * trainable_params / max(all_params, 1)):.4f}%)."
        )

    def _resolve_lora_init_path(self, init_path: str) -> str:
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

    def save_pretrained(self, save_directory: str) -> None:
        """Export current training state to a Diffusers-compatible checkpoint folder."""
        save_dir = Path(save_directory)
        save_dir.mkdir(parents=True, exist_ok=True)

        if self.lora_enabled:
            lora_dir = save_dir / "unet_lora"
            self.unet.save_lora_adapter(str(lora_dir), adapter_name=self.lora_adapter_name)

            export_payload = {
                "base_model_name_or_path": self.cfg.get("pretrained_model_name_or_path"),
                "adapter_name": self.lora_adapter_name,
                "lora": self.lora_cfg,
            }

            save_json_config(export_payload, str(save_dir / "lora_config.json"))
            save_json_config(self.cfg, str(save_dir / "training_config.json"))
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
        pipeline.save_pretrained(str(save_dir), safe_serialization=True)

        save_json_config(self.cfg, str(save_dir / "training_config.json"))

    def save_hf_checkpoint(self, checkpoint_filepath: str) -> None:
        """Export UNet-style HF checkpoint colocated with a Lightning .ckpt filepath."""
        checkpoint_dir = Path(checkpoint_filepath).with_suffix("")
        hf_dir = checkpoint_dir / "hf_checkpoint"
        hf_dir.mkdir(parents=True, exist_ok=True)

        if self.lora_enabled:
            lora_dir = hf_dir / "unet_lora"
            self.unet.save_lora_adapter(str(lora_dir), adapter_name=self.lora_adapter_name)

            export_payload = {
                "base_model_name_or_path": self.cfg.get("pretrained_model_name_or_path"),
                "adapter_name": self.lora_adapter_name,
                "lora": self.lora_cfg,
            }

            save_json_config(export_payload, str(hf_dir / "lora_config.json"))
            save_json_config(self.cfg, str(hf_dir / "training_config.json"))
            return

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

        save_json_config(self.cfg, str(hf_dir / "training_config.json"))

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
            [param for param in self.unet.parameters() if param.requires_grad],
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
