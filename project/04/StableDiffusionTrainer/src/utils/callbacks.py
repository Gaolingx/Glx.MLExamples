from typing import Any, Dict, List, Optional
from pathlib import Path

import torch
import pytorch_lightning as pl
from pytorch_lightning import Callback, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.rank_zero import rank_zero_info, rank_zero_only
from diffusers.optimization import get_scheduler

from src.utils.config import save_json_config


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
            rank_zero_info(
                f"[NaNLossCallback] Non-finite loss detected at global_step={trainer.global_step}. Stopping training.")


class LRandSchedulerOverrideCallback(Callback):
    """Override optimizer LR and optionally reset scheduler state after ckpt resume."""

    def __init__(self, cfg: Dict[str, Any]) -> None:
        super().__init__()
        self.training_cfg = cfg.get("training", {})
        self.applied = False

    def on_fit_start(self, trainer: Trainer, pl_module: pl.LightningModule) -> None:
        if self.applied:
            return

        reset_lr = bool(self.training_cfg.get("override_lr_on_resume", False))
        reset_scheduler = bool(self.training_cfg.get("reset_scheduler_on_resume", False))

        if not reset_lr and not reset_scheduler:
            return

        if not trainer.optimizers:
            return

        optimizer = trainer.optimizers[0]
        target_lr = float(self.training_cfg.get("learning_rate", 1e-4))

        if reset_lr:
            for group in optimizer.param_groups:
                group["lr"] = target_lr
                group["initial_lr"] = target_lr
            rank_zero_info(f"[LRandSchedulerOverrideCallback] Reset optimizer LR to {target_lr}.")

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
            rank_zero_info(
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
            hf_dir = Path(filepath).parent / "hf_checkpoint"
            pl_module.save_hf_checkpoint(hf_dir)


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

        pl_module.log("train/grad_norm", grad_norm, on_step=True, on_epoch=False, prog_bar=False, sync_dist=True)

    def on_before_zero_grad(
            self,
            trainer: pl.Trainer,
            pl_module: pl.LightningModule,
            optimizer: torch.optim.Optimizer) -> None:
        step = int(trainer.global_step)
        if step == 0 or step % self.log_every_n_steps != 0:
            return

        grad_norm = self._compute_global_norm(pl_module, use_grad=True).detach().cpu()

        pl_module.log("train/grad_norm_clip", grad_norm, on_step=True, on_epoch=False, prog_bar=False, sync_dist=True)


class LoggingCallback(Callback):
    """Lightweight stdout logging callback for training progress."""

    def __init__(self, log_every_n_steps: int = 10) -> None:
        super().__init__()
        self.log_every_n_steps = max(1, int(log_every_n_steps))

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
            "train/loss",
            "train/loss_ema",
            "train/lr",
        }

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

        metrics: Dict[str, Any] = {}
        runtime_metrics = getattr(pl_module, "runtime_log_dict", None)
        if isinstance(runtime_metrics, dict):
            metrics.update(runtime_metrics)

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


class ConfigSnapshotCallback(Callback):
    """Persist the resolved experiment config next to checkpoints."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config

    @rank_zero_only
    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Write the experiment configuration once at the start of training."""

        output_dir = Path(trainer.default_root_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        config_path = output_dir / "resolved_config.json"
        save_json_config(self.config, config_path)
