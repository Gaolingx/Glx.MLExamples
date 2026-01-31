"""
Custom PyTorch Lightning callbacks for VAE training.
Includes image logging, checkpoint management, and training monitoring.
"""

from typing import Optional, Dict, Any
from pathlib import Path
import json

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
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

        # Get images
        with torch.no_grad():
            targets = batch["pixel_values"][: self.num_images]
            reconstructions, _ = pl_module(targets, sample_posterior=False)

        # Denormalize
        targets = (targets + 1) / 2
        reconstructions = (reconstructions + 1) / 2

        # Clamp
        targets = torch.clamp(targets, 0, 1)
        reconstructions = torch.clamp(reconstructions, 0, 1)

        # Create comparison grid
        comparison = torch.cat([targets, reconstructions], dim=0)
        grid = torchvision.utils.make_grid(comparison, nrow=self.num_images, padding=2)

        # Log to tensorboard
        if hasattr(trainer.logger, "experiment"):
            trainer.logger.experiment.add_image(
                "train/comparison", grid, trainer.global_step
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

            # Save VAE
            if hasattr(trainer.lightning_module, "vae"):
                trainer.lightning_module.vae.save_pretrained(str(hf_dir))

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
        total_norm = total_norm**0.5

        pl_module.log(
            "train/grad_norm",
            total_norm,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
        )
