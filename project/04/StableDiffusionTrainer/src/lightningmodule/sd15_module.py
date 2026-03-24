import importlib
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from packaging import version
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from diffusers.utils.import_utils import is_xformers_available
from pytorch_lightning.utilities.rank_zero import rank_zero_info, rank_zero_warn
from huggingface_hub import create_repo, upload_folder
from transformers import CLIPTextModel, CLIPTokenizer
from transformers.modeling_outputs import BaseModelOutputWithPooling

from src.utils.config import load_json_config
from src.utils.train_utils import (
    resolve_lora_init_path,
    log_validation_images,
    run_validation_inference,
    build_inference_scheduler,
)


class StableDiffusionLightningModule(pl.LightningModule):
    """Lightning module for SD1.5 denoising training in latent space."""

    def __init__(self, cfg: Dict[str, Any]) -> None:
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters(cfg)
        self.lora_cfg = cfg.get("lora", {})
        self.lora_enabled = bool(self.lora_cfg.get("enabled", False))
        self.lora_adapter_name = str(self.lora_cfg.get("adapter_name", "default"))
        self.train_unet_lora = bool(self.lora_cfg.get("train_unet", self.lora_enabled))
        self.train_text_encoder_lora = bool(self.lora_cfg.get("train_text_encoder", False))
        self.model_card_path: Optional[Path] = None

        model_name = cfg.get("pretrained_model_name_or_path")
        clip_model_name = cfg.get("clip_model_name_or_path", "CompVis/stable-diffusion-v1-4")
        vae_model_name = cfg.get("vae_model_name_or_path", "stabilityai/sd-vae-ft-mse")
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
            self.vae = AutoencoderKL.from_pretrained(vae_model_name, subfolder="vae")

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
            self.text_encoder.requires_grad_(False)

        self.training_cfg = cfg["training"]
        self.validation_cfg = cfg.get("validation", {})
        self.clip_cfg = cfg.get("clip", {})
        self.clip_skip = max(1, int(self.clip_cfg.get("skip", 1)))

        # Runtime metric cache for callback-side unified stdout logging.
        self.runtime_log_dict: Dict[str, float] = {}
        self.loss_ema: Optional[float] = None

        if bool(self.training_cfg.get("gradient_checkpointing", False)):
            self.unet.enable_gradient_checkpointing()
            if self.train_text_encoder_lora:
                self.text_encoder.gradient_checkpointing_enable()

        if bool(self.training_cfg.get("enable_xformers_memory_efficient_attention", False)):
            if is_xformers_available():
                import xformers

                xformers_version = version.parse(xformers.__version__)
                if xformers_version == version.parse("0.0.16"):
                    rank_zero_warn(
                        "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, "
                        "please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                    )
                self.unet.enable_xformers_memory_efficient_attention()
            else:
                raise ValueError("xformers is not available. Make sure it is installed correctly")

    def _setup_lora(self) -> None:
        try:
            peft_module = importlib.import_module("peft")
            LoraConfig = peft_module.LoraConfig
        except ImportError as exc:
            raise ImportError("LoRA training requires `peft`. Please install it with `pip install peft`.") from exc

        if not self.cfg.get("pretrained_model_name_or_path"):
            raise ValueError("`pretrained_model_name_or_path` must be provided when `lora.enabled=true`.")

        init_path = self.lora_cfg.get("init_path")
        resolved_init_path = resolve_lora_init_path(init_path) if init_path else None
        use_dora = bool(self.lora_cfg.get("use_dora", False))
        target_modules = self.lora_cfg.get("target_modules", ["to_q", "to_k", "to_v", "to_out.0"])

        if self.train_unet_lora:
            if resolved_init_path is None:
                lora_config = LoraConfig(
                    r=int(self.lora_cfg.get("rank", 16)),
                    lora_alpha=int(self.lora_cfg.get("alpha", 16)),
                    lora_dropout=float(self.lora_cfg.get("dropout", 0.0)),
                    bias=str(self.lora_cfg.get("bias", "none")),
                    target_modules=target_modules,
                    init_lora_weights=self.lora_cfg.get("init_lora_weights", True),
                    use_dora=use_dora,
                )
                self.unet.add_adapter(lora_config, adapter_name=self.lora_adapter_name)
                rank_zero_info(
                    f"[LoRA] Added UNet adapter '{self.lora_adapter_name}' with rank={lora_config.r}, alpha={lora_config.lora_alpha}, dora={use_dora}."
                )
            else:
                self.unet.load_lora_adapter(
                    resolved_init_path,
                    prefix=None,
                    adapter_name=self.lora_adapter_name,
                )
                rank_zero_info(f"[LoRA] Loaded UNet adapter '{self.lora_adapter_name}' from {resolved_init_path}.")

            self.unet.set_adapter(self.lora_adapter_name)
            self.unet.requires_grad_(False)
            for name, parameter in self.unet.named_parameters():
                parameter.requires_grad = "lora_" in name
        else:
            self.unet.requires_grad_(False)

        if self.train_text_encoder_lora:
            text_encoder_targets = self.lora_cfg.get("text_encoder_target_modules", ["q_proj", "k_proj", "v_proj", "out_proj"])
            lora_config = LoraConfig(
                r=int(self.lora_cfg.get("rank", 16)),
                lora_alpha=int(self.lora_cfg.get("alpha", 16)),
                lora_dropout=float(self.lora_cfg.get("dropout", 0.0)),
                bias=str(self.lora_cfg.get("bias", "none")),
                target_modules=text_encoder_targets,
                init_lora_weights=self.lora_cfg.get("init_lora_weights", True),
                use_dora=use_dora,
            )
            self.text_encoder.add_adapter(lora_config)
            self.text_encoder.requires_grad_(False)
            for name, parameter in self.text_encoder.named_parameters():
                parameter.requires_grad = "lora_" in name
            rank_zero_info(
                f"[LoRA] Added text encoder adapter with rank={lora_config.r}, alpha={lora_config.lora_alpha}, dora={use_dora}."
            )

        trainable_params = sum(param.numel() for param in self.unet.parameters() if param.requires_grad)
        all_params = sum(param.numel() for param in self.unet.parameters())
        rank_zero_info(
            f"[LoRA] Training {trainable_params:,} / {all_params:,} UNet parameters "
            f"({(100.0 * trainable_params / max(all_params, 1)):.4f}%)."
        )
        if self.train_text_encoder_lora:
            text_trainable = sum(param.numel() for param in self.text_encoder.parameters() if param.requires_grad)
            text_total = sum(param.numel() for param in self.text_encoder.parameters())
            rank_zero_info(
                f"[LoRA] Training {text_trainable:,} / {text_total:,} text encoder parameters "
                f"({(100.0 * text_trainable / max(text_total, 1)):.4f}%)."
            )

    def on_fit_start(self) -> None:
        # Keep frozen modules in eval mode during training.
        self.vae.eval()
        if self.train_text_encoder_lora:
            self.text_encoder.train()
        else:
            self.text_encoder.eval()

    def encode_prompt(self, input_ids: torch.Tensor) -> torch.Tensor:
        if self.train_text_encoder_lora:
            return self._encode_prompt_impl(input_ids)

        with torch.no_grad():
            return self._encode_prompt_impl(input_ids)

    def _encode_prompt_impl(self, input_ids: torch.Tensor) -> torch.Tensor:
        if self.clip_skip <= 1:
            return self.text_encoder(input_ids, attention_mask=None)[0]

        text_encoder_output = self.text_encoder(
            input_ids,
            attention_mask=None,
            output_hidden_states=True,
            return_dict=True,
        )
        return self._select_text_encoder_hidden_state(text_encoder_output)

    def _select_text_encoder_hidden_state(
            self,
            text_encoder_output: BaseModelOutputWithPooling,
    ) -> torch.Tensor:
        hidden_states = text_encoder_output.hidden_states
        if hidden_states is None:
            raise ValueError("CLIP hidden states are unavailable; cannot apply clip skip.")

        max_skip = len(hidden_states) - 1
        if self.clip_skip > max_skip:
            raise ValueError(
                f"clip.skip={self.clip_skip} exceeds available hidden-state depth ({max_skip})."
            )

        selected_hidden_state = hidden_states[-self.clip_skip]
        final_layer_norm = getattr(self.text_encoder.text_model, "final_layer_norm", None)
        if final_layer_norm is not None:
            selected_hidden_state = final_layer_norm(selected_hidden_state)
        return selected_hidden_state

    def _compute_snr(self, timesteps: torch.Tensor) -> torch.Tensor:
        alphas_cumprod = self.noise_scheduler.alphas_cumprod.to(device=timesteps.device, dtype=torch.float32)
        alpha = alphas_cumprod[timesteps]
        sigma = (1.0 - alpha).clamp(min=1e-8)
        return alpha / sigma

    def _encode_pixel_values(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # VAE expects [B, C, H, W]. If an upstream batch of size 1 was flattened
        # to [C, H, W], recover the missing batch dimension.
        if pixel_values.ndim == 3:
            pixel_values = pixel_values.unsqueeze(0)
        elif pixel_values.ndim != 4:
            raise ValueError(f"Expected pixel_values to be 4D [B, C, H, W], got shape {tuple(pixel_values.shape)}")

        pixel_values = pixel_values.to(device=self.device, dtype=self.vae.dtype)

        latents = self.vae.encode(pixel_values).latent_dist.sample()
        return latents * self.vae.config.scaling_factor

    def _get_latents_from_batch(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, float]:
        use_cached_latent = batch.get("use_cached_latent")
        latent_batch = batch.get("latent")

        pixel_values = batch.get("pixel_values")
        if latent_batch is None or use_cached_latent is None:
            if pixel_values is None:
                raise ValueError("Batch must contain either cached `latent` values or `pixel_values`.")
            return self._encode_pixel_values(pixel_values), 0.0

        if isinstance(use_cached_latent, torch.Tensor):
            cached_mask = use_cached_latent.detach().cpu().bool().tolist()
        else:
            cached_mask = [bool(flag) for flag in use_cached_latent]

        if isinstance(latent_batch, torch.Tensor):
            cached_latents = latent_batch
        else:
            cached_latents = torch.stack(latent_batch)

        if cached_latents.ndim == 3:
            cached_latents = cached_latents.unsqueeze(0)

        cached_latents = cached_latents.to(device=self.device, dtype=self.unet.dtype)
        cache_hit_rate = float(sum(cached_mask)) / float(len(cached_mask)) if cached_mask else 0.0

        if all(cached_mask):
            return cached_latents, cache_hit_rate

        if pixel_values is None:
            raise ValueError("Batch must contain either cached `latent` values or `pixel_values`.")

        if pixel_values.ndim == 3:
            pixel_values = pixel_values.unsqueeze(0)

        pixel_values = pixel_values.to(device=self.device, dtype=self.vae.dtype)
        latents = torch.empty_like(cached_latents)

        uncached_indices = [idx for idx, is_cached in enumerate(cached_mask) if not is_cached]
        cached_indices = [idx for idx, is_cached in enumerate(cached_mask) if is_cached]

        if cached_indices:
            latents[cached_indices] = cached_latents[cached_indices]

        if uncached_indices:
            uncached_pixel_values = pixel_values[uncached_indices]
            uncached_latents = self._encode_pixel_values(uncached_pixel_values).to(dtype=self.unet.dtype)
            latents[uncached_indices] = uncached_latents

        return latents, cache_hit_rate

    def _compute_loss(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, Any]]:
        input_ids = batch["input_ids"]

        if input_ids.ndim == 3:
            input_ids = input_ids.squeeze(1)
        elif input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)

        latents, cache_hit_rate = self._get_latents_from_batch(batch)

        noise = torch.randn_like(latents)
        noise_offset = float(self.training_cfg.get("noise_offset", 0.0))
        if noise_offset != 0.0:
            noise = noise + noise_offset * torch.randn(
                (latents.shape[0], latents.shape[1], 1, 1),
                device=latents.device,
                dtype=latents.dtype,
            )
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

        snr = self._compute_snr(timesteps)
        loss_per_sample = F.mse_loss(model_pred.float(), target.float(), reduction="none")
        loss_per_sample = loss_per_sample.mean(dim=tuple(range(1, loss_per_sample.ndim)))

        snr_gamma = self.training_cfg.get("snr_gamma")
        if snr_gamma is not None:
            gamma = float(snr_gamma)
            snr_weights = torch.minimum(snr, torch.full_like(snr, gamma)) / snr.clamp(min=1e-8)
            if self.noise_scheduler.config.prediction_type == "v_prediction":
                snr_weights = snr_weights + 1.0
            loss = (loss_per_sample * snr_weights).mean()
        else:
            loss = loss_per_sample.mean()

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
            "train/latent_cache_hit_rate": float(cache_hit_rate),
        }
        if snr_gamma is not None:
            metrics["train/snr_gamma"] = float(snr_gamma)
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

        global_step = int(self.global_step)
        every_n_steps = int(self.validation_cfg.get("every_n_steps", 0))
        if global_step == 0 or every_n_steps <= 0 or global_step % every_n_steps != 0:
            return

        prompts = self.validation_cfg.get("prompts", [])
        if len(prompts) == 0:
            return

        images = run_validation_inference(
            vae=self.vae,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            unet=self.unet,
            device=self.device,
            clip_skip=self.clip_skip,
            validation_cfg=self.validation_cfg,
            scheduler=build_inference_scheduler(
                self.noise_scheduler.config,
                self.validation_cfg.get("sampler"),
                self.validation_cfg.get("scheduler"),
            ),
        )

        log_validation_images(self.logger, images, global_step)

    def save_pretrained(self, save_directory: str) -> None:
        """Export current training state to a Diffusers-compatible checkpoint folder."""
        save_dir = Path(save_directory)
        save_dir.mkdir(parents=True, exist_ok=True)

        if self.lora_enabled:
            StableDiffusionPipeline.save_lora_weights(
                save_directory=str(save_dir),
                unet_lora_layers=self.unet.get_adapter_state_dict() if self.train_unet_lora else None,
                text_encoder_lora_layers=(
                    self.text_encoder.get_adapter_state_dict() if self.train_text_encoder_lora else None
                ),
                weight_name="pytorch_lora_weights.safetensors",
                safe_serialization=True,
            )
            return

        pipeline = StableDiffusionPipeline(
            vae=self.vae,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            unet=self.unet,
            scheduler=build_inference_scheduler(
                self.noise_scheduler.config,
                self.validation_cfg.get("sampler"),
                self.validation_cfg.get("scheduler"),
            ),
            safety_checker=None,
            feature_extractor=None,
            requires_safety_checker=False,
        )
        pipeline.save_pretrained(str(save_dir), safe_serialization=True)

    def save_hf_checkpoint(self, checkpoint_filepath: str) -> None:
        """Export UNet-style HF checkpoint colocated with a Lightning .ckpt filepath."""
        hf_dir = Path(checkpoint_filepath)
        hf_dir.mkdir(parents=True, exist_ok=True)

        if self.lora_enabled:
            if self.train_unet_lora:
                lora_dir = hf_dir / "unet_lora"
                self.unet.save_lora_adapter(str(lora_dir), adapter_name=self.lora_adapter_name)
            if self.train_text_encoder_lora:
                text_lora_dir = hf_dir / "text_encoder_lora"
                self.text_encoder.save_pretrained(str(text_lora_dir), safe_serialization=True)

            return

        unet_save_dir = hf_dir / "unet"
        self.unet.save_pretrained(str(unet_save_dir))

    def on_train_end(self) -> None:
        if self.trainer is None or not self.trainer.is_global_zero:
            return

        hub_cfg = self.cfg.get("hub", {})
        if not bool(hub_cfg.get("push_to_hub", False)):
            return

        output_dir = Path(self.cfg.get("output_dir", self.trainer.default_root_dir))
        export_dir = output_dir / ("final_lora" if self.lora_enabled else "final_model")
        project_root = Path(__file__).resolve().parents[2]
        source_readme = project_root / "README.md"

        def ensure_hub_readme() -> None:
            if source_readme.exists():
                (export_dir / "README.md").write_text(source_readme.read_text(encoding="utf-8"), encoding="utf-8")

        if not export_dir.is_dir():
            rank_zero_info(f"[Hub] Export directory `{export_dir}` is missing or incomplete. Re-exporting before upload.")
            self.save_pretrained(str(export_dir))
            ensure_hub_readme()

        token = hub_cfg.get("token") or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        repo_id = hub_cfg.get("model_id")
        if not repo_id:
            rank_zero_warn("[Hub] `hub.model_id` is missing, skipping upload.")
            return

        repo_id = str(repo_id)
        create_repo(repo_id=repo_id, token=token, exist_ok=True)

        upload_folder(
            repo_id=repo_id,
            folder_path=str(export_dir),
            token=token,
            commit_message="Upload Diffusers model artifacts",
        )
        rank_zero_info(f"[Hub] Uploaded artifacts to https://huggingface.co/{repo_id}")

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

        trainable_parameters = [param for param in self.unet.parameters() if param.requires_grad]
        trainable_parameters.extend(param for param in self.text_encoder.parameters() if param.requires_grad)

        optimizer = optimizer_cls(
            trainable_parameters,
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
