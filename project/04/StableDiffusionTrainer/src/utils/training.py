import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pytorch_lightning as pl
import torch
from diffusers.optimization import get_scheduler
from pytorch_lightning import Callback, Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.utilities.rank_zero import rank_zero_info
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
            rank_zero_info(f"[NaNLossCallback] Non-finite loss detected at global_step={trainer.global_step}. Stopping training.")


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
        monitor=checkpoint_cfg.get("monitor", "train/loss"),
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
        LoggingCallback(log_every_n_steps=int(logging_cfg.get("log_every_n_steps", 10))),
    ]


def find_resume_checkpoint(resume_arg: str, default_ckpt_dir: str) -> Optional[str]:
    """Resolve a checkpoint path for resuming training."""
    if not resume_arg:
        return None

    if resume_arg.lower() == "last":
        last_ckpt = Path(default_ckpt_dir) / "last.ckpt"
        if last_ckpt.exists():
            print(f"Resuming from latest checkpoint: {last_ckpt}")
            return str(last_ckpt)
        return None

    p = Path(resume_arg)
    if p.is_file() and p.suffix == ".ckpt":
        print(f"Resuming from checkpoint file: {p}")
        return str(p)
    if p.is_dir():
        candidates = sorted(
            [x for x in p.rglob("*.ckpt") if x.name != "last.ckpt"],
            key=lambda x: x.stat().st_mtime,
            reverse=True,
        )
        if candidates:
            print(f"Resuming from checkpoint in dir: {candidates[0]}")
            return str(candidates[0])

        last_ckpt = p / "last.ckpt"
        if last_ckpt.exists():
            print(f"Resuming from last checkpoint in dir: {last_ckpt}")
            return str(last_ckpt)
    return None


def build_trainer_kwargs(config: dict) -> Dict[str, Any]:
    """Build Trainer kwargs with optional DDP/multi-GPU support."""
    distributed_config = config.get("distributed", {})

    accelerator = distributed_config.get("accelerator", "auto")
    devices: Union[str, int, List[int]] = distributed_config.get("devices", "auto")
    strategy: Union[str, DDPStrategy] = distributed_config.get("strategy", "auto")
    num_nodes = int(distributed_config.get("num_nodes", 1))

    if isinstance(strategy, str) and strategy.lower() == "ddp":
        strategy = DDPStrategy(
            find_unused_parameters=distributed_config.get("find_unused_parameters", False),
            gradient_as_bucket_view=distributed_config.get("gradient_as_bucket_view", True),
        )

    trainer_kwargs = {
        "accelerator": accelerator,
        "devices": devices,
        "strategy": strategy,
        "num_nodes": num_nodes,
    }

    if distributed_config.get("sync_batchnorm", False):
        trainer_kwargs["sync_batchnorm"] = True

    return trainer_kwargs


def seed_everything(seed: int) -> None:
    pl.seed_everything(seed, workers=True)
