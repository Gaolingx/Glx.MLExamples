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
    Override LR and/or reset scheduler state when resuming from checkpoint.

    Supports:
    1) Override LR only, keep scheduler progress
    2) Override LR + reset scheduler (restart warmup/progress)
    3) Reset scheduler only

    Notes:
    - Designed for diffusers.optimization.get_scheduler (typically LambdaLR),
      but includes fallbacks for non-Lambda schedulers.
    """

    def __init__(
            self,
            override_lr_on_resume: bool = True,
            reset_scheduler_on_resume: bool = False,
            gen_opt_config: Optional[Dict[str, Any]] = None,
            disc_opt_config: Optional[Dict[str, Any]] = None,
            verbose: bool = True,
    ):
        super().__init__()
        self.override_lr_on_resume = override_lr_on_resume
        self.reset_scheduler_on_resume = reset_scheduler_on_resume
        self.gen_opt_config = gen_opt_config or {}
        self.disc_opt_config = disc_opt_config or {}
        self.verbose = verbose

    # ----------------------------
    # Public hook
    # ----------------------------
    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if not self._is_resuming(trainer):
            self._log("Starting fresh training, no LR/scheduler override applied.")
            return

        if not self.override_lr_on_resume and not self.reset_scheduler_on_resume:
            self._log("Resuming from checkpoint, but both override/reset are disabled.")
            return

        self._log(f"Resuming from checkpoint: {trainer.ckpt_path}")

        optimizers = self._as_list(getattr(trainer, "optimizers", []))
        scheduler_configs = self._as_list(getattr(trainer, "lr_scheduler_configs", []))
        schedulers = [
            cfg.scheduler for cfg in scheduler_configs
            if hasattr(cfg, "scheduler")
        ]

        role_configs: List[Tuple[str, Dict[str, Any]]] = [
            ("VAE", self.gen_opt_config),
            ("Discriminator", self.disc_opt_config),
        ]

        num_slots = max(len(optimizers), len(schedulers), len(role_configs))
        for idx in range(num_slots):
            name, opt_cfg = role_configs[idx] if idx < len(role_configs) else (f"Optimizer_{idx}", {})
            optimizer = optimizers[idx] if idx < len(optimizers) else None
            scheduler = schedulers[idx] if idx < len(schedulers) else None
            target_lr = opt_cfg.get("learning_rate")

            self._process_one(name, optimizer, scheduler, target_lr)

    # ----------------------------
    # Core per-optimizer logic
    # ----------------------------
    def _process_one(
            self,
            name: str,
            optimizer: Optional[torch.optim.Optimizer],
            scheduler: Optional[Any],
            target_lr: Optional[float],
    ) -> None:
        if optimizer is None:
            self._log(f"{name}: optimizer not found, skip.")
            return

        # 1) Reset scheduler state if requested
        if self.reset_scheduler_on_resume and scheduler is not None:
            self._reset_scheduler_state(name, scheduler)

        # 2) Override LR if requested
        if self.override_lr_on_resume and target_lr is not None:
            self._override_optimizer_and_scheduler_lr(
                name=name,
                optimizer=optimizer,
                scheduler=scheduler,
                new_lr=float(target_lr),
                reset_mode=self.reset_scheduler_on_resume,
            )
            return

        # 3) Reset-only (no LR override): re-derive current lr from reset scheduler
        if self.reset_scheduler_on_resume and scheduler is not None:
            reset_lrs = self._compute_reset_current_lrs(scheduler, optimizer)
            self._apply_optimizer_lrs(optimizer, reset_lrs)
            self._set_scheduler_last_lrs(scheduler, reset_lrs)
            self._step_scheduler_to_epoch_zero(name, scheduler)
            self._log(f"{name}: scheduler reset only, synced optimizer lr -> {reset_lrs}")

    def _override_optimizer_and_scheduler_lr(
            self,
            name: str,
            optimizer: torch.optim.Optimizer,
            scheduler: Optional[Any],
            new_lr: float,
            reset_mode: bool,
    ) -> None:
        n_groups = len(optimizer.param_groups)
        new_base_lrs = [new_lr] * n_groups

        old_current_lrs = [float(pg.get("lr", 0.0)) for pg in optimizer.param_groups]

        if scheduler is not None:
            old_base_lrs = self._get_base_lrs(scheduler, optimizer)

            if reset_mode:
                multipliers = self._get_reset_multipliers(scheduler, n_groups)
            else:
                multipliers = self._compute_progress_multipliers(
                    old_current_lrs=old_current_lrs,
                    old_base_lrs=old_base_lrs,
                    n_groups=n_groups,
                )

            self._set_scheduler_base_lrs(scheduler, new_base_lrs)
        else:
            multipliers = [1.0] * n_groups

        new_current_lrs = [b * m for b, m in zip(new_base_lrs, multipliers)]

        # Apply to optimizer param groups
        for pg, base_lr, cur_lr in zip(optimizer.param_groups, new_base_lrs, new_current_lrs):
            pg["initial_lr"] = float(base_lr)
            pg["lr"] = float(cur_lr)

        # Sync scheduler internal bookkeeping
        if scheduler is not None:
            self._set_scheduler_last_lrs(scheduler, new_current_lrs)
            if reset_mode:
                self._step_scheduler_to_epoch_zero(name, scheduler)

        self._log(
            f"{name}: override lr -> base={new_lr:.3e}, "
            f"current={new_current_lrs}, reset_mode={reset_mode}"
        )

    # ----------------------------
    # Scheduler state helpers
    # ----------------------------
    def _reset_scheduler_state(self, name: str, scheduler: Any) -> None:
        """Reset scheduler progress counters back to initial state."""
        old_last_epoch = getattr(scheduler, "last_epoch", None)

        if hasattr(scheduler, "last_epoch"):
            scheduler.last_epoch = -1
        if hasattr(scheduler, "_step_count"):
            scheduler._step_count = 0
        if hasattr(scheduler, "_get_lr_called_within_step"):
            scheduler._get_lr_called_within_step = False

        self._log(
            f"{name}: reset scheduler {type(scheduler).__name__} "
            f"(last_epoch: {old_last_epoch} -> -1)"
        )

    def _step_scheduler_to_epoch_zero(self, name: str, scheduler: Any) -> None:
        try:
            scheduler.step()
            self._log(
                f"{name}: stepped scheduler to epoch 0 "
                f"(last_epoch={getattr(scheduler, 'last_epoch', '?')})"
            )
        except Exception as e:
            self._log(f"{name}: failed to step scheduler to epoch 0: {e}")

    def _compute_reset_current_lrs(
            self,
            scheduler: Any,
            optimizer: torch.optim.Optimizer,
    ) -> List[float]:
        base_lrs = self._get_base_lrs(scheduler, optimizer)
        multipliers = self._get_reset_multipliers(scheduler, len(base_lrs))
        return [b * m for b, m in zip(base_lrs, multipliers)]

    def _get_reset_multipliers(self, scheduler: Any, n_groups: int) -> List[float]:
        """Compute the LR multiplier at step 0 (post-reset)."""
        if isinstance(scheduler, LambdaLR) and hasattr(scheduler, "lr_lambdas"):
            vals = []
            for fn in scheduler.lr_lambdas:
                try:
                    vals.append(float(fn(0)))
                except Exception:
                    vals.append(1.0)
            return self._match_len(vals, n_groups, fill=1.0)

        return [1.0] * n_groups

    def _compute_progress_multipliers(
            self,
            old_current_lrs: List[float],
            old_base_lrs: List[float],
            n_groups: int,
    ) -> List[float]:
        cur = self._match_len(old_current_lrs, n_groups, fill=0.0)
        base = self._match_len(old_base_lrs, n_groups, fill=0.0)

        multipliers = []
        eps = 1e-12
        for c, b in zip(cur, base):
            if abs(b) < eps:
                multipliers.append(1.0)
            else:
                multipliers.append(float(c / b))
        return multipliers

    def _get_base_lrs(
            self,
            scheduler: Optional[Any],
            optimizer: torch.optim.Optimizer,
    ) -> List[float]:
        if scheduler is not None and hasattr(scheduler, "base_lrs"):
            base = list(scheduler.base_lrs)
            return self._match_len(base, len(optimizer.param_groups), fill=base[-1] if len(base) > 0 else 0.0)

        vals = []
        for pg in optimizer.param_groups:
            vals.append(float(pg.get("initial_lr", pg.get("lr", 0.0))))
        return vals

    def _set_scheduler_base_lrs(self, scheduler: Any, base_lrs: List[float]) -> None:
        if hasattr(scheduler, "base_lrs"):
            scheduler.base_lrs = list(base_lrs)

    def _set_scheduler_last_lrs(self, scheduler: Any, lrs: List[float]) -> None:
        if hasattr(scheduler, "_last_lr"):
            scheduler._last_lr = list(lrs)

    def _apply_optimizer_lrs(
            self,
            optimizer: torch.optim.Optimizer,
            lrs: List[float],
    ) -> None:
        for pg, lr in zip(optimizer.param_groups, lrs):
            pg["lr"] = float(lr)

    # ----------------------------
    # Utility helpers
    # ----------------------------
    @staticmethod
    def _as_list(x: Any) -> List[Any]:
        if x is None:
            return []
        if isinstance(x, list):
            return x
        return [x]

    @staticmethod
    def _match_len(values: List[float], n: int, fill: float = 0.0) -> List[float]:
        if len(values) >= n:
            return values[:n]
        if len(values) == 0:
            return [fill] * n
        return values + [values[-1]] * (n - len(values))

    @staticmethod
    def _is_resuming(trainer: pl.Trainer) -> bool:
        return getattr(trainer, "ckpt_path", None) is not None

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"[LRandSchedulerOverrideCallback] {msg}")


class NaNLossCallback(Callback):
    """
    Immediately stop training when a NaN or Inf loss is detected.

    Checks every training step output for non-finite values and raises
    a KeyboardInterrupt to cleanly halt the Trainer.
    """

    def on_train_batch_end(
            self,
            trainer: pl.Trainer,
            pl_module: pl.LightningModule,
            outputs: Any,
            batch: Any,
            batch_idx: int,
    ) -> None:
        # outputs may be a Tensor, a dict with key "loss", or None
        loss = None
        if isinstance(outputs, torch.Tensor):
            loss = outputs
        elif isinstance(outputs, dict) and "loss" in outputs:
            loss = outputs["loss"]

        if loss is not None and not torch.isfinite(loss):
            print(
                f"\n[NaNLossCallback] Non-finite loss detected at "
                f"step {trainer.global_step} (batch_idx={batch_idx}): {loss.item():.6f}. "
                f"Stopping training."
            )
            trainer.should_stop = True
