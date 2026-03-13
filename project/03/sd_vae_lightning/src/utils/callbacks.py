"""
Custom PyTorch Lightning callbacks for VAE training.
Includes image logging, checkpoint management, and training monitoring.
"""

from typing import Optional, Dict, Any, Tuple, List
from pathlib import Path
import json

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from pytorch_lightning.utilities.types import STEP_OUTPUT
from pytorch_lightning.utilities.rank_zero import rank_zero_info
import torchvision
import math


class VAELoggingCallback(Callback):
    """
    Unified callback for VAE metric/image logging.

    - Logs step-level train metrics emitted by `training_step`
    - Aggregates and logs epoch-level train metrics
    - Logs validation metrics emitted by `validation_step`
    - Computes/logs validation rFID at epoch end
    - Logs validation comparison images to TensorBoard
    """

    def __init__(
            self,
            num_val_images: int = 4,
            log_to_tensorboard: bool = True,
            log_images_every_n_steps: int = 500,
    ):
        super().__init__()
        self.num_val_images = num_val_images
        self.log_to_tensorboard = log_to_tensorboard
        self.log_images_every_n_steps = log_images_every_n_steps
        self._val_outputs: List[Dict[str, torch.Tensor]] = []

    @staticmethod
    def _format_metric_value(value: Any) -> Optional[float]:
        if value is None:
            return None
        if torch.is_tensor(value):
            if value.numel() != 1:
                return None
            return float(value.detach().float().cpu().item())
        return float(value)

    @staticmethod
    def _to_epoch_metric_name(metric_name: str) -> Optional[str]:
        if not metric_name.startswith("train/"):
            return None
        return metric_name.replace("train/", "epoch/", 1)

    @staticmethod
    def _is_prog_bar_metric(metric_name: str) -> bool:
        return metric_name in {
            "train/rec_loss",
            "train/nll_loss",
            "train/lr",
            "train/disc_lr",
        }

    def _visualize_latent(
            self,
            latent: torch.Tensor,
            target_size: Tuple[int, int],
    ) -> torch.Tensor:
        b, c, h, w = latent.shape

        latent_flat = latent.view(b, -1)
        min_vals = latent_flat.min(dim=1, keepdim=True)[0].view(b, 1, 1, 1)
        max_vals = latent_flat.max(dim=1, keepdim=True)[0].view(b, 1, 1, 1)

        range_vals = max_vals - min_vals
        range_vals = torch.where(range_vals < 1e-5, torch.ones_like(range_vals), range_vals)
        latent_normalized = (latent - min_vals) / range_vals

        if c >= 3:
            latent_vis = latent_normalized[:, :3]
        else:
            latent_vis = latent_normalized[:, :1].repeat(1, 3, 1, 1)

        latent_vis = F.interpolate(
            latent_vis,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )
        return latent_vis

    def _log_images(
            self,
            logger: pl.Trainer.logger,
            targets: torch.Tensor,
            reconstructions: torch.Tensor,
            latent: torch.Tensor,
            prefix: str,
            step: int,
    ) -> None:

        targets = torch.clamp((targets + 1) / 2, 0, 1)
        reconstructions = torch.clamp((reconstructions + 1) / 2, 0, 1)
        target_size = (targets.shape[2], targets.shape[3])
        latent_vis = self._visualize_latent(latent, target_size)

        n = targets.shape[0]
        comparison = torch.cat([targets[:n], latent_vis[:n], reconstructions[:n]], dim=0)
        grid = torchvision.utils.make_grid(comparison, nrow=n, padding=2)

        if hasattr(logger, "experiment"):
            logger.experiment.add_image(f"{prefix}/reconstruction", grid, step)

    def on_train_batch_end(
            self,
            trainer: pl.Trainer,
            pl_module: pl.LightningModule,
            outputs: Any,
            batch: Any,
            batch_idx: int,
    ) -> None:
        metrics: Dict[str, Any] = {}
        if isinstance(outputs, dict):
            output_metrics = outputs.get("train_metrics", {})
            if isinstance(output_metrics, dict):
                metrics.update(output_metrics)

        if not metrics:
            return

        for key, value in metrics.items():
            metric_value = self._format_metric_value(value)
            if metric_value is None:
                continue

            train_key = key if key.startswith("train/") else f"train/{key}"
            epoch_key = self._to_epoch_metric_name(train_key)
            if epoch_key is not None:
                pl_module.log(
                    epoch_key,
                    metric_value,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                    sync_dist=True,
                )

            pl_module.log(
                train_key,
                metric_value,
                on_step=True,
                on_epoch=False,
                prog_bar=self._is_prog_bar_metric(train_key),
                sync_dist=True,
            )

        if (
                self.log_to_tensorboard
                and trainer.logger is not None
                and trainer.is_global_zero
                and trainer.global_step % self.log_images_every_n_steps == 0
                and isinstance(batch, dict)
                and "pixel_values" in batch
        ):
            with torch.no_grad():
                targets = batch["pixel_values"][: self.num_val_images]
                posterior = pl_module.vae.encode(targets).latent_dist
                latent = posterior.mode()
                reconstructions = pl_module.vae.decode(latent).sample

            self._log_images(trainer.logger, targets, reconstructions, latent, "train", trainer.global_step)

    def on_validation_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._val_outputs = []

    def on_validation_batch_end(
            self,
            trainer: pl.Trainer,
            pl_module: pl.LightningModule,
            outputs: Optional[STEP_OUTPUT],
            batch: Any,
            batch_idx: int,
            dataloader_idx: int = 0,
    ) -> None:
        if not isinstance(outputs, dict):
            return

        metrics = outputs.get("val_metrics", {})
        if isinstance(metrics, dict):
            for key, value in metrics.items():
                metric_value = self._format_metric_value(value)
                pl_module.log(
                    f"val/{key}",
                    metric_value,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                    sync_dist=True,
                )

        visuals = outputs.get("val_visuals")
        if isinstance(visuals, dict) and len(self._val_outputs) < self.num_val_images:
            self._val_outputs.append(visuals)

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if hasattr(pl_module, "rfid_metric"):
            rfid_score = pl_module.rfid_metric.compute()
            pl_module.log(
                "val/rfid",
                rfid_score,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True
            )
            pl_module.rfid_metric.reset()

        if (
                self.log_to_tensorboard
                and trainer.logger is not None
                and trainer.is_global_zero
                and len(self._val_outputs) > 0
        ):
            targets = torch.cat([out["targets"] for out in self._val_outputs], dim=0)
            reconstructions = torch.cat([out["reconstructions"] for out in self._val_outputs], dim=0)
            latent = torch.cat([out["latent"] for out in self._val_outputs], dim=0)

            n = min(self.num_val_images, targets.shape[0])
            targets = targets[:n]
            reconstructions = reconstructions[:n]
            latent = latent[:n]
            self._log_images(trainer.logger, targets, reconstructions, latent, "val", trainer.global_step)


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

        if self.save_hf_format:
            # Save HuggingFace format
            hf_dir = Path(filepath).parent / "hf_checkpoint"
            trainer.lightning_module.save_hf_checkpoint(hf_dir)


class LRandSchedulerOverrideCallback(Callback):
    """Override optimizer LR and optionally rebuild schedulers when resuming from a checkpoint."""

    def __init__(
            self,
            override_lr_on_resume: bool = False,
            reset_scheduler_on_resume: bool = False,
            gen_opt_config: Optional[Dict[str, Any]] = None,
            disc_opt_config: Optional[Dict[str, Any]] = None,
            verbose: bool = True,
    ) -> None:
        super().__init__()
        self.override_lr_on_resume = override_lr_on_resume
        self.reset_scheduler_on_resume = reset_scheduler_on_resume
        self.gen_opt_config = gen_opt_config or {}
        self.disc_opt_config = disc_opt_config or {}
        self.verbose = verbose
        self.applied = False

    def _log(self, msg: str) -> None:
        if self.verbose:
            rank_zero_info(f"[LRandSchedulerOverrideCallback] {msg}")

    def _is_resume_run(self, trainer: pl.Trainer) -> bool:
        return bool(getattr(trainer, "ckpt_path", None))

    @staticmethod
    def _compute_total_steps(trainer: pl.Trainer, pl_module: pl.LightningModule) -> int:
        estimated_steps = trainer.estimated_stepping_batches
        if estimated_steps is None or not math.isfinite(estimated_steps):
            raise ValueError("trainer.estimated_stepping_batches is None or infinite; cannot rebuild LR scheduler.")

        accumulate_grad_batches = max(1, int(getattr(pl_module, "accumulate_grad_batches", 1)))
        return max(1, int(math.ceil(estimated_steps / accumulate_grad_batches)))

    @staticmethod
    def _iter_scheduler_configs_for_optimizer(
            trainer: pl.Trainer,
            optimizer: torch.optim.Optimizer,
    ):
        for scheduler_config in trainer.lr_scheduler_configs:
            scheduler = scheduler_config.scheduler
            if getattr(scheduler, "optimizer", None) is optimizer:
                yield scheduler_config

    def _override_optimizer_lr(
            self,
            trainer: pl.Trainer,
            optimizer: torch.optim.Optimizer,
            opt_config: Dict[str, Any],
            optimizer_name: str,
    ) -> None:
        target_lr = float(opt_config.get("learning_rate", 1e-4))

        for group in optimizer.param_groups:
            group["lr"] = target_lr
            group["initial_lr"] = target_lr

        for scheduler_config in self._iter_scheduler_configs_for_optimizer(trainer, optimizer):
            scheduler = scheduler_config.scheduler
            if hasattr(scheduler, "base_lrs"):
                scheduler.base_lrs = [target_lr for _ in optimizer.param_groups]
            if hasattr(scheduler, "_last_lr"):
                scheduler._last_lr = [target_lr for _ in optimizer.param_groups]

        self._log(f"Override {optimizer_name} optimizer LR to {target_lr}.")

    def _reset_scheduler(
            self,
            trainer: pl.Trainer,
            pl_module: pl.LightningModule,
            optimizer: torch.optim.Optimizer,
            opt_config: Dict[str, Any],
            scheduler_idx: int,
            scheduler_name: str,
            total_steps: int,
    ) -> None:
        new_scheduler = pl_module._build_scheduler(
            optimizer=optimizer,
            opt_config=opt_config,
            num_training_steps=max(1, total_steps),
            last_epoch=-1,
        )

        if scheduler_idx < len(trainer.lr_scheduler_configs):
            trainer.lr_scheduler_configs[scheduler_idx].scheduler = new_scheduler

        scheduler_type = opt_config.get("lr_scheduler", "constant_with_warmup")
        self._log(
            f"Reset scheduler for {scheduler_name} "
            f"(type={scheduler_type}, total_steps={total_steps})."
        )

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if self.applied:
            return

        if not self._is_resume_run(trainer):
            return

        if not self.override_lr_on_resume and not self.reset_scheduler_on_resume:
            return

        if not trainer.optimizers:
            return

        total_steps = self._compute_total_steps(trainer, pl_module)

        if len(trainer.optimizers) >= 1 and self.gen_opt_config:
            if self.override_lr_on_resume:
                self._override_optimizer_lr(trainer, trainer.optimizers[0], self.gen_opt_config, "generator")
            if self.reset_scheduler_on_resume and trainer.lr_scheduler_configs:
                self._reset_scheduler(
                    trainer,
                    pl_module,
                    trainer.optimizers[0],
                    self.gen_opt_config,
                    scheduler_idx=0,
                    scheduler_name="generator",
                    total_steps=total_steps,
                )

        if len(trainer.optimizers) >= 2 and self.disc_opt_config:
            disc_total_steps = total_steps
            if getattr(pl_module, "use_discriminator", False):
                disc_start_step = int(getattr(pl_module, "disc_start_step", 0))
                disc_total_steps = max(1, total_steps - disc_start_step)

            if self.override_lr_on_resume:
                self._override_optimizer_lr(trainer, trainer.optimizers[1], self.disc_opt_config, "discriminator")
            if self.reset_scheduler_on_resume and len(trainer.lr_scheduler_configs) >= 2:
                self._reset_scheduler(
                    trainer,
                    pl_module,
                    trainer.optimizers[1],
                    self.disc_opt_config,
                    scheduler_idx=1,
                    scheduler_name="discriminator",
                    total_steps=disc_total_steps,
                )

        if self.override_lr_on_resume:
            if hasattr(pl_module, "learning_rate") and self.gen_opt_config:
                pl_module.learning_rate = float(self.gen_opt_config.get("learning_rate", pl_module.learning_rate))
            if hasattr(pl_module, "disc_learning_rate") and self.disc_opt_config:
                pl_module.disc_learning_rate = float(
                    self.disc_opt_config.get("learning_rate", pl_module.disc_learning_rate)
                )

        self.applied = True


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
            rank_zero_info(
                f"\n[NaNLossCallback] Non-finite loss detected at "
                f"step {trainer.global_step} (batch_idx={batch_idx}): {loss.item():.6f}. "
                f"Stopping training."
            )
            trainer.should_stop = True
