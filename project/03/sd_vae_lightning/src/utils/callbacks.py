"""
Custom PyTorch Lightning callbacks for VAE training.
Includes image logging, checkpoint management, and training monitoring.
"""

from typing import Optional, Dict, Any, Tuple, List
from pathlib import Path
import json
import os

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from torch.optim.lr_scheduler import LambdaLR
import torchvision


class ImageLoggerCallback(Callback):
    """
    Callback for logging reconstructed images during training.

    Args:
        log_every_n_steps: Log images every N training steps.
        num_images: Number of images to log.
        log_to_tensorboard: Whether to log to TensorBoard.
    """

    def __init__(
            self,
            log_every_n_steps: int = 500,
            num_images: int = 4,
            log_to_tensorboard: bool = True,
    ):
        super().__init__()
        self.log_every_n_steps = log_every_n_steps
        self.num_images = num_images
        self.log_to_tensorboard = log_to_tensorboard

    def _visualize_latent(
            self,
            latent: torch.Tensor,
            target_size: Tuple[int, int],
    ) -> torch.Tensor:
        """
        Visualize latent space as an image.

        Args:
            latent: Latent tensor of shape (B, C, H, W).
            target_size: Target size (H, W) to upsample to.

        Returns:
            Visualization tensor of shape (B, 3, H', W') normalized to [0, 1].
        """
        b, c, h, w = latent.shape

        # Min-max normalize per image (across all channels)
        latent_flat = latent.view(b, -1)
        min_vals = latent_flat.min(dim=1, keepdim=True)[0].view(b, 1, 1, 1)
        max_vals = latent_flat.max(dim=1, keepdim=True)[0].view(b, 1, 1, 1)

        # Avoid division by zero
        range_vals = max_vals - min_vals
        range_vals = torch.where(range_vals < 1e-5, torch.ones_like(range_vals), range_vals)

        latent_normalized = (latent - min_vals) / range_vals

        # Use first 3 channels as RGB if available, otherwise use grayscale
        if c >= 3:
            latent_vis = latent_normalized[:, :3]
        else:
            # Repeat single channel to RGB
            latent_vis = latent_normalized[:, :1].repeat(1, 3, 1, 1)

        # Upsample to target size for consistent grid visualization
        latent_vis = F.interpolate(
            latent_vis,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )

        return latent_vis

    def on_train_batch_end(
            self,
            trainer: pl.Trainer,
            pl_module: pl.LightningModule,
            outputs: Any,
            batch: Dict[str, torch.Tensor],
            batch_idx: int,
    ) -> None:
        """Log images at specified intervals during training."""
        if trainer.global_step % self.log_every_n_steps != 0:
            return

        if not self.log_to_tensorboard or trainer.logger is None:
            return

        # Get images and latent
        with torch.no_grad():
            targets = batch["pixel_values"][: self.num_images]

            # Use forward_with_latent if available, otherwise use standard forward and encode
            if hasattr(pl_module, "forward_with_latent"):
                reconstructions, latent, _ = pl_module.forward_with_latent(targets, sample_posterior=False)
            else:
                reconstructions, _ = pl_module(targets, sample_posterior=False)
                # Get latent separately
                latent = pl_module.vae.encode(targets).latent_dist.mode()

        # Denormalize
        targets = (targets + 1) / 2
        reconstructions = (reconstructions + 1) / 2

        # Clamp
        targets = torch.clamp(targets, 0, 1)
        reconstructions = torch.clamp(reconstructions, 0, 1)

        # Visualize latent space
        target_size = (targets.shape[2], targets.shape[3])
        latent_vis = self._visualize_latent(latent, target_size)

        # Create comparison grid with 3 rows: targets, latent, reconstructions
        comparison = torch.cat([targets, latent_vis, reconstructions], dim=0)
        grid = torchvision.utils.make_grid(comparison, nrow=self.num_images, padding=2)

        # Log to tensorboard
        if hasattr(trainer.logger, "experiment"):
            trainer.logger.experiment.add_image(
                "batch/comparison", grid, trainer.global_step
            )


class VAECheckpointCallback(ModelCheckpoint):
    """
    Extended ModelCheckpoint for VAE models.
    Saves both Lightning checkpoint and HuggingFace-compatible weights.

    Args:
        dirpath: Directory to save checkpoints.
        filename: Checkpoint filename template.
        save_top_k: Number of best checkpoints to keep.
        monitor: Metric to monitor for best checkpoint.
        mode: 'min' or 'max' for monitored metric.
        save_last: Whether to save the last checkpoint.
        every_n_train_steps: Save every N training steps.
        save_hf_format: Whether to save in HuggingFace format.
    """

    def __init__(
            self,
            dirpath: Optional[str] = None,
            filename: str = "vae-{epoch:02d}-{step:06d}",
            save_top_k: int = 3,
            monitor: str = "val/rec_loss",
            mode: str = "min",
            save_last: bool = True,
            every_n_train_steps: Optional[int] = None,
            save_hf_format: bool = True,
    ):
        super().__init__(
            dirpath=dirpath,
            filename=filename,
            save_top_k=save_top_k,
            monitor=monitor,
            mode=mode,
            save_last=save_last,
            every_n_train_steps=every_n_train_steps,
            save_weights_only=False,
            verbose=True,
        )
        self.save_hf_format = save_hf_format

    def _save_checkpoint(self, trainer: pl.Trainer, filepath: str) -> None:
        """Save checkpoint in both Lightning and HuggingFace formats."""
        super()._save_checkpoint(trainer, filepath)

        if self.save_hf_format and trainer.is_global_zero:
            # Save HuggingFace format
            hf_dir = Path(filepath).parent / "hf_checkpoint"
            hf_dir.mkdir(parents=True, exist_ok=True)

            pl_module = trainer.lightning_module
            vae_save_dir = hf_dir / "vae"
            pl_module.vae.save_pretrained(str(vae_save_dir))

            if getattr(pl_module, "use_ema", False) and getattr(pl_module, "ema", None) is not None:
                ema_save_dir = hf_dir / "vae_ema"
                print(f"Saving EMA weights to {ema_save_dir}...")

                pl_module.ema.save_pretrained(str(ema_save_dir))

            # Save training config
            config_path = hf_dir / "training_config.json"
            with open(config_path, "w") as f:
                json.dump(trainer.lightning_module.config, f, indent=2)


class LearningRateMonitor(Callback):
    """
    Callback for monitoring and logging learning rates.
    """

    def on_train_batch_start(
            self,
            trainer: pl.Trainer,
            pl_module: pl.LightningModule,
            batch: Any,
            batch_idx: int,
    ) -> None:
        """Log current learning rates."""
        if trainer.logger is None:
            return

        optimizers = trainer.optimizers
        if not isinstance(optimizers, list):
            optimizers = [optimizers]

        for idx, optimizer in enumerate(optimizers):
            for param_group_idx, param_group in enumerate(optimizer.param_groups):
                lr = param_group["lr"]
                name = f"lr/optimizer_{idx}_group_{param_group_idx}"
                pl_module.log(name, lr, on_step=True, on_epoch=False)


class GradientNormLogger(Callback):
    """
    Callback for logging gradient norms during training.
    Useful for debugging training instabilities.
    """

    def __init__(self, log_every_n_steps: int = 100):
        super().__init__()
        self.log_every_n_steps = log_every_n_steps

    def on_before_optimizer_step(
            self,
            trainer: pl.Trainer,
            pl_module: pl.LightningModule,
            optimizer: torch.optim.Optimizer,
    ) -> None:
        """Log gradient norms before optimizer step."""
        if trainer.global_step % self.log_every_n_steps != 0:
            return

        total_norm = 0.0
        for p in pl_module.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5

        pl_module.log(
            "train/grad_norm",
            total_norm,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
        )


class LRandSchedulerOverrideCallback(Callback):
    """
    Callback to override learning rate and/or reset scheduler state when resuming from checkpoint.

    Designed to work with diffusers.optimization.get_scheduler (returns LambdaLR).

    This is useful when you want to:
    - Resume training with a different learning rate than what was saved in the checkpoint
    - Reset the scheduler to start fresh (e.g., restart warmup) while keeping model weights

    Args:
        override_lr_on_resume: If True, override optimizer LR with config values after loading checkpoint.
        reset_scheduler_on_resume: If True, reset scheduler state (step counter) to initial state.
        vae_lr: Learning rate for VAE optimizer (used when override_lr_on_resume=True).
        disc_lr: Learning rate for discriminator optimizer (used when override_lr_on_resume=True).
        verbose: Whether to print info messages about overrides.
    """

    def __init__(
            self,
            override_lr_on_resume: bool = True,
            reset_scheduler_on_resume: bool = False,
            vae_lr: Optional[float] = None,
            disc_lr: Optional[float] = None,
            verbose: bool = True,
    ):
        super().__init__()
        self.override_lr_on_resume = override_lr_on_resume
        self.reset_scheduler_on_resume = reset_scheduler_on_resume
        self.vae_lr = vae_lr
        self.disc_lr = disc_lr
        self.verbose = verbose

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """
        Called when training starts. If resuming from checkpoint, apply LR/scheduler overrides.
        """
        # Check if we're resuming from a checkpoint
        if trainer.ckpt_path is None:
            if self.verbose:
                print("[LRandSchedulerOverrideCallback] Starting fresh training, no overrides applied.")
            return

        if self.verbose:
            print(f"[LRandSchedulerOverrideCallback] Resuming from checkpoint: {trainer.ckpt_path}")

        # Get optimizers
        optimizers = trainer.optimizers
        if not isinstance(optimizers, list):
            optimizers = [optimizers]

        # Get scheduler configs
        lr_scheduler_configs = trainer.lr_scheduler_configs
        if lr_scheduler_configs is None:
            lr_scheduler_configs = []

        # Reset scheduler state FIRST if enabled (before LR override)
        if self.reset_scheduler_on_resume:
            self._reset_schedulers(lr_scheduler_configs, optimizers)

        # Override learning rates if enabled
        if self.override_lr_on_resume:
            self._override_learning_rates(optimizers, lr_scheduler_configs, pl_module)

    def _override_learning_rates(
            self,
            optimizers: List[torch.optim.Optimizer],
            lr_scheduler_configs: List[Any],
            pl_module: pl.LightningModule,
    ) -> None:
        """Override optimizer learning rates AND scheduler base_lrs with config values."""
        # Get LR values from config or use provided values
        config = getattr(pl_module, "config", {})
        train_config = config.get("training", {})

        vae_lr = self.vae_lr or train_config.get("learning_rate")
        disc_lr = self.disc_lr or train_config.get("disc_learning_rate")

        target_lrs = [vae_lr, disc_lr]
        names = ["VAE", "Discriminator"]

        for idx, optimizer in enumerate(optimizers):
            if idx >= len(target_lrs) or target_lrs[idx] is None:
                continue

            new_lr = target_lrs[idx]
            old_lr = optimizer.param_groups[0]["lr"]

            # Update optimizer param groups
            for param_group in optimizer.param_groups:
                param_group["lr"] = new_lr
                # Update initial_lr (used by schedulers as reference)
                param_group["initial_lr"] = new_lr

            # Update scheduler's base_lrs
            if idx < len(lr_scheduler_configs):
                scheduler = lr_scheduler_configs[idx].scheduler
                self._update_scheduler_base_lr(scheduler, new_lr, names[idx])

            if self.verbose:
                print(f"[LRandSchedulerOverrideCallback] {names[idx]} optimizer LR: {old_lr:.2e} -> {new_lr:.2e}")

    def _update_scheduler_base_lr(
            self,
            scheduler: Any,
            new_lr: float,
            name: str,
    ) -> None:
        """
        Update scheduler's base learning rate.

        Works with LambdaLR from diffusers.optimization.get_scheduler.
        """
        # Update base_lrs - the reference LR that lambda functions multiply
        if hasattr(scheduler, "base_lrs"):
            old_base_lrs = list(scheduler.base_lrs)
            scheduler.base_lrs = [new_lr for _ in scheduler.base_lrs]
            if self.verbose:
                print(
                    f"[LRandSchedulerOverrideCallback] {name} scheduler base_lrs: "
                    f"{old_base_lrs} -> {scheduler.base_lrs}"
                )

        # Update _last_lr to reflect the change immediately
        if hasattr(scheduler, "_last_lr") and hasattr(scheduler, "base_lrs"):
            current_step = max(0, getattr(scheduler, "last_epoch", 0))

            # For LambdaLR (diffusers get_scheduler returns this type)
            if isinstance(scheduler, LambdaLR) and hasattr(scheduler, "lr_lambdas"):
                scheduler._last_lr = [
                    lmbda(current_step) * base_lr
                    for lmbda, base_lr in zip(scheduler.lr_lambdas, scheduler.base_lrs)
                ]
            else:
                # Fallback for other scheduler types
                scheduler._last_lr = list(scheduler.base_lrs)

    def _reset_schedulers(
            self,
            lr_scheduler_configs: List[Any],
            optimizers: List[torch.optim.Optimizer],
    ) -> None:
        """
        Reset scheduler state to initial values.

        Works with LambdaLR from diffusers.optimization.get_scheduler.
        """
        for idx, scheduler_config in enumerate(lr_scheduler_configs):
            scheduler = scheduler_config.scheduler
            scheduler_name = type(scheduler).__name__
            old_last_epoch = getattr(scheduler, "last_epoch", -1)

            # Reset last_epoch to -1 (will become 0 on first step())
            if hasattr(scheduler, "last_epoch"):
                scheduler.last_epoch = -1

            # Reset _step_count if it exists (used internally by some schedulers)
            if hasattr(scheduler, "_step_count"):
                scheduler._step_count = 0

            # Recalculate _last_lr for step 0 (initial state after first step)
            if hasattr(scheduler, "_last_lr") and hasattr(scheduler, "base_lrs"):
                if isinstance(scheduler, LambdaLR) and hasattr(scheduler, "lr_lambdas"):
                    # LambdaLR: _last_lr = lambda(step) * base_lr
                    # At step 0 (after reset, first step will compute lambda(0))
                    scheduler._last_lr = [
                        lmbda(0) * base_lr
                        for lmbda, base_lr in zip(scheduler.lr_lambdas, scheduler.base_lrs)
                    ]
                else:
                    scheduler._last_lr = list(scheduler.base_lrs)

            # Also reset initial_lr in optimizer param_groups to match base_lrs
            if idx < len(optimizers) and hasattr(scheduler, "base_lrs"):
                optimizer = optimizers[idx]
                for pg_idx, param_group in enumerate(optimizer.param_groups):
                    if pg_idx < len(scheduler.base_lrs):
                        param_group["initial_lr"] = scheduler.base_lrs[pg_idx]

            if self.verbose:
                current_lr = scheduler._last_lr[0] if hasattr(scheduler, "_last_lr") else "N/A"
                print(
                    f"[LRandSchedulerOverrideCallback] Scheduler {idx} ({scheduler_name}): "
                    f"last_epoch {old_last_epoch} -> -1 (reset), initial LR -> {current_lr}"
                )
