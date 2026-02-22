import json
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
    """ModelCheckpoint with Lightning ckpt + HuggingFace-style UNet export."""

    def __init__(self, *args: Any, save_hf_format: bool = True, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.save_hf_format = save_hf_format

    @staticmethod
    def _extract_training_config(pl_module: pl.LightningModule) -> Dict[str, Any]:
        if hasattr(pl_module, "cfg") and isinstance(pl_module.cfg, dict):
            return pl_module.cfg
        if hasattr(pl_module, "config") and isinstance(pl_module.config, dict):
            return pl_module.config
        return {}

    @staticmethod
    def _save_ema_unet(pl_module: pl.LightningModule, hf_dir: Path) -> None:
        use_ema = bool(getattr(pl_module, "use_ema", False))
        if not use_ema:
            return

        ema_obj = getattr(pl_module, "ema", None)
        ema_unet = getattr(pl_module, "ema_unet", None)
        ema_save_dir = hf_dir / "unet_ema"

        # Compatibility branch: either an EMA wrapper exposing save_pretrained,
        # or a standalone ema_unet module.
        if ema_obj is not None and hasattr(ema_obj, "save_pretrained"):
            ema_obj.save_pretrained(str(ema_save_dir))
            return

        if ema_unet is not None and hasattr(ema_unet, "save_pretrained"):
            ema_unet.save_pretrained(str(ema_save_dir))

    def _save_checkpoint(self, trainer: Trainer, filepath: str) -> None:
        super()._save_checkpoint(trainer, filepath)

        if not self.save_hf_format or not trainer.is_global_zero:
            return

        pl_module = trainer.lightning_module

        # Keep HF export colocated with each ckpt stem:
        # xxx-epoch=00-step=xxxxxx-val/hf_checkpoint
        checkpoint_dir = Path(filepath).with_suffix("")
        hf_dir = checkpoint_dir / "hf_checkpoint"
        hf_dir.mkdir(parents=True, exist_ok=True)

        unet_save_dir = hf_dir / "unet"
        pl_module.unet.save_pretrained(str(unet_save_dir))
        self._save_ema_unet(pl_module, hf_dir)

        config_path = hf_dir / "training_config.json"
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(self._extract_training_config(pl_module), f, indent=2, ensure_ascii=False)


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

        runtime_metrics = getattr(pl_module, "runtime_log_dict", {})
        loss = runtime_metrics.get("train/loss")
        lr = runtime_metrics.get("train/lr")

        # Backward-compatible fallback: keep prior behavior if runtime dict is unavailable.
        if loss is None or lr is None:
            metrics = trainer.callback_metrics
            loss = metrics.get("train/loss_step") or metrics.get("train/loss")
            lr = metrics.get("train/lr")

        if trainer.is_global_zero:
            loss_val = float(loss.detach().cpu()) if torch.is_tensor(loss) else float(loss) if loss is not None else float("nan")
            lr_val = float(lr.detach().cpu()) if torch.is_tensor(lr) else float(lr) if lr is not None else float("nan")
            print(f"[LoggingCallback] step={step} train/loss={loss_val:.6f} train/lr={lr_val:.8f}")


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
