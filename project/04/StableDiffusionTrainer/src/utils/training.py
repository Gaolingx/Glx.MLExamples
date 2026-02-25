import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

import pytorch_lightning as pl
import torch
from diffusers.optimization import get_scheduler
from pytorch_lightning import Callback, Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger


class NaNLossCallback(Callback):
    """Stop training immediately when NaN/Inf loss is detected."""

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        loss_tensor: Optional[torch.Tensor] = None

        if torch.is_tensor(outputs):
            loss_tensor = outputs.detach()
        elif isinstance(outputs, dict) and "loss" in outputs and torch.is_tensor(outputs["loss"]):
            loss_tensor = outputs["loss"].detach()

        if loss_tensor is None:
            return

        if not torch.isfinite(loss_tensor).all():
            trainer.should_stop = True
            if trainer.is_global_zero:
                print(f"[NaNLossCallback] Non-finite loss detected at global_step={trainer.global_step}. Stopping training.")


class LRandSchedulerOverrideCallback(Callback):
    """Override optimizer LR and optionally reset scheduler state after ckpt resume."""

    def __init__(self, cfg: Dict[str, Any]) -> None:
        super().__init__()
        self.training_cfg = cfg.get("training", {})
        self.override_cfg = cfg.get("resume_override", {})
        self.applied = False

    def on_fit_start(self, trainer: Trainer, pl_module: pl.LightningModule) -> None:
        if self.applied:
            return

        reset_lr = bool(self.override_cfg.get("reset_lr", False))
        reset_scheduler = bool(self.override_cfg.get("reset_scheduler", False))

        if not reset_lr and not reset_scheduler:
            return

        if not trainer.optimizers:
            return

        optimizer = trainer.optimizers[0]
        target_lr = float(self.override_cfg.get("learning_rate", self.training_cfg.get("learning_rate", 1e-4)))

        if reset_lr:
            for group in optimizer.param_groups:
                group["lr"] = target_lr
                group["initial_lr"] = target_lr
            if trainer.is_global_zero:
                print(f"[LRandSchedulerOverrideCallback] Reset optimizer LR to {target_lr}.")

        if reset_scheduler:
            scheduler_type = self.training_cfg.get("lr_scheduler", "cosine")
            warmup_steps = int(self.training_cfg.get("lr_warmup_steps", 0))
            total_steps = max(1, trainer.estimated_stepping_batches)

            new_scheduler = get_scheduler(
                scheduler_type,
                optimizer=optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps,
            )

            if trainer.lr_scheduler_configs:
                trainer.lr_scheduler_configs[0].scheduler = new_scheduler
            if trainer.is_global_zero:
                print(
                    "[LRandSchedulerOverrideCallback] Reset scheduler state "
                    f"(type={scheduler_type}, warmup={warmup_steps}, total_steps={total_steps})."
                )

        self.applied = True


class CheckpointCallback(ModelCheckpoint):
    """ModelCheckpoint with optional HF export delegated to LightningModule."""

    def __init__(self, *args: Any, save_hf_format: bool = True, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.save_hf_format = save_hf_format

    def _save_checkpoint(self, trainer: Trainer, filepath: str) -> None:
        super()._save_checkpoint(trainer, filepath)

        if not self.save_hf_format or not trainer.is_global_zero:
            return

        pl_module = trainer.lightning_module
        if hasattr(pl_module, "save_hf_checkpoint"):
            pl_module.save_hf_checkpoint(filepath)


class GradParamNormCallback(Callback):
    """Log global parameter/gradient L2 norms as training metrics."""

    def __init__(self, log_every_n_steps: int = 1) -> None:
        super().__init__()
        self.log_every_n_steps = max(1, int(log_every_n_steps))

    @staticmethod
    def _compute_global_norm(pl_module: pl.LightningModule, *, use_grad: bool) -> torch.Tensor:
        reference = None
        total = None

        for param in pl_module.parameters():
            if not param.requires_grad:
                continue

            tensor = param.grad if use_grad else param.detach()
            if tensor is None:
                continue

            reference = tensor
            part = tensor.detach().float().pow(2).sum()
            total = part if total is None else total + part

        if total is None:
            if reference is not None:
                return torch.tensor(0.0, device=reference.device)
            return torch.tensor(0.0, device=pl_module.device)

        return total.sqrt()

    def on_after_backward(self, trainer: Trainer, pl_module: pl.LightningModule) -> None:
        step = int(trainer.global_step)
        if step == 0 or step % self.log_every_n_steps != 0:
            return

        grad_norm = self._compute_global_norm(pl_module, use_grad=True).detach().cpu()
        param_norm = self._compute_global_norm(pl_module, use_grad=False).detach().cpu()

        pl_module.log("train/grad_norm", grad_norm, on_step=True, on_epoch=False, prog_bar=False)
        pl_module.log("train/param_norm", param_norm, on_step=True, on_epoch=False, prog_bar=False)

    def on_before_optimizer_step(
            self,
            trainer: pl.Trainer,
            pl_module: pl.LightningModule,
            optimizer: torch.optim.Optimizer) -> None:
        step = int(trainer.global_step)
        if step == 0 or step % self.log_every_n_steps != 0:
            return

        grad_norm = self._compute_global_norm(pl_module, use_grad=True).detach().cpu()

        pl_module.log("train/grad_norm_clip", grad_norm, on_step=True, on_epoch=False, prog_bar=False)


class LoggingCallback(Callback):
    """Lightweight stdout logging callback for training progress."""

    def __init__(self, log_every_n_steps: int = 10) -> None:
        super().__init__()
        self.log_every_n_steps = max(1, int(log_every_n_steps))

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        step = int(trainer.global_step)
        if step == 0 or step % self.log_every_n_steps != 0:
            return

        if not isinstance(outputs, dict):
            return

        metrics = outputs.get("train_metrics", {})
        if not isinstance(metrics, dict):
            return

        for key, value in metrics.items():
            if value is None:
                continue
            if torch.is_tensor(value):
                metric_value = value.detach()
            else:
                metric_value = float(value)
            pl_module.log(key, metric_value, on_step=True, on_epoch=False, prog_bar=True)


class TrainHealthMetricsCallback(Callback):
    """Log timing/throughput and latent/noise/timestep distribution sanity metrics."""

    def __init__(self, log_every_n_steps: int = 10) -> None:
        super().__init__()
        self.log_every_n_steps = max(1, int(log_every_n_steps))
        self._batch_start_time: Optional[float] = None

    @staticmethod
    def _infer_batch_size(batch: Any) -> int:
        if torch.is_tensor(batch):
            return int(batch.shape[0]) if batch.ndim > 0 else 1

        if isinstance(batch, dict):
            for value in batch.values():
                if torch.is_tensor(value):
                    return int(value.shape[0]) if value.ndim > 0 else 1

        if isinstance(batch, (list, tuple)) and len(batch) > 0:
            first = batch[0]
            if torch.is_tensor(first):
                return int(first.shape[0]) if first.ndim > 0 else 1

        return 1

    @staticmethod
    def _to_device(value: torch.Tensor, device: torch.device) -> torch.Tensor:
        if value.device == device:
            return value
        return value.to(device)

    def on_train_batch_start(
        self,
        trainer: Trainer,
        pl_module: pl.LightningModule,
        batch: Any,
        batch_idx: int,
    ) -> None:
        self._batch_start_time = time.perf_counter()

    @torch.no_grad()
    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        step = int(trainer.global_step)
        if step == 0 or step % self.log_every_n_steps != 0:
            return

        # Throughput/system-level metrics.
        if self._batch_start_time is not None:
            step_time_ms = (time.perf_counter() - self._batch_start_time) * 1000.0
            batch_size = max(1, self._infer_batch_size(batch))
            samples_per_sec = (batch_size * 1000.0) / max(step_time_ms, 1e-6)

            pl_module.log("train/step_time_ms", float(step_time_ms), on_step=True, on_epoch=False, prog_bar=True)
            pl_module.log("train/samples_per_sec", float(samples_per_sec), on_step=True, on_epoch=False, prog_bar=True)

        # Sanity metrics for diffusion training internals.
        if not isinstance(batch, dict) or "pixel_values" not in batch:
            return

        if not hasattr(pl_module, "vae") or not hasattr(pl_module, "noise_scheduler"):
            return

        pixel_values = batch["pixel_values"]
        if not torch.is_tensor(pixel_values):
            return

        if pixel_values.ndim == 3:
            pixel_values = pixel_values.unsqueeze(0)
        elif pixel_values.ndim != 4:
            return

        pixel_values = self._to_device(pixel_values, pl_module.device)

        latents = pl_module.vae.encode(pixel_values).latent_dist.sample()
        latents = latents * pl_module.vae.config.scaling_factor

        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        timesteps = torch.randint(
            0,
            pl_module.noise_scheduler.config.num_train_timesteps,
            (bsz,),
            device=latents.device,
            dtype=torch.long,
        )

        timestep_mean = timesteps.float().mean()
        timestep_std = timesteps.float().std(unbiased=False)
        latent_std = latents.float().std(unbiased=False)
        noise_std = noise.float().std(unbiased=False)

        pl_module.log("train/timestep_mean", float(timestep_mean.detach().cpu()), on_step=True, on_epoch=False, prog_bar=True)
        pl_module.log("train/timestep_std", float(timestep_std.detach().cpu()), on_step=True, on_epoch=False, prog_bar=True)
        pl_module.log("train/latent_std", float(latent_std.detach().cpu()), on_step=True, on_epoch=False, prog_bar=True)
        pl_module.log("train/noise_std", float(noise_std.detach().cpu()), on_step=True, on_epoch=False, prog_bar=True)


def build_tensorboard_logger(logging_cfg: Dict[str, Any]) -> TensorBoardLogger:
    return TensorBoardLogger(
        save_dir=logging_cfg.get("tensorboard_save_dir", "outputs/tb_logs"),
        name=logging_cfg.get("name", "sd15_lightning"),
        default_hp_metric=False,
    )


def build_callbacks(cfg: Dict[str, Any]) -> list:
    checkpoint_cfg = cfg.get("checkpoint", {})
    logging_cfg = cfg.get("logging", {})

    ckpt_callback = CheckpointCallback(
        dirpath=checkpoint_cfg.get("dirpath", "outputs/checkpoints"),
        filename="sd15-epoch={epoch:02d}-step={step:06d}-val",
        monitor=checkpoint_cfg.get("monitor", "train/loss_epoch"),
        mode=checkpoint_cfg.get("mode", "min"),
        save_top_k=int(checkpoint_cfg.get("save_top_k", 3)),
        save_last=bool(checkpoint_cfg.get("save_last", True)),
        every_n_train_steps=int(checkpoint_cfg.get("every_n_train_steps", 100)),
        auto_insert_metric_name=False,
    )

    return [
        ckpt_callback,
        LearningRateMonitor(logging_interval="step"),
        NaNLossCallback(),
        LRandSchedulerOverrideCallback(cfg),
        GradParamNormCallback(log_every_n_steps=int(logging_cfg.get("norm_log_every_n_steps", 1))),
        TrainHealthMetricsCallback(log_every_n_steps=int(logging_cfg.get("health_log_every_n_steps", 10))),
        LoggingCallback(log_every_n_steps=int(logging_cfg.get("log_every_n_steps", 10))),
    ]


def find_resume_checkpoint(checkpoint_cfg: Dict[str, Any]) -> Optional[str]:
    ckpt_dir = Path(checkpoint_cfg.get("dirpath", "outputs/checkpoints"))
    if not ckpt_dir.exists():
        return None

    last_ckpt = ckpt_dir / "last.ckpt"
    if last_ckpt.exists():
        print(f"Resuming from latest checkpoint: {last_ckpt}")
        return str(last_ckpt)

    candidates = sorted(ckpt_dir.glob("*.ckpt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if len(candidates) == 0:
        return None

    return str(candidates[0])


def seed_everything(seed: int) -> None:
    pl.seed_everything(seed, workers=True)
