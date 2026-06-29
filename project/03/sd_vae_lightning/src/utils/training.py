"""
Utility functions for building Lightning Trainer, callbacks, and logging.
"""
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    EarlyStopping,
    RichProgressBar,
)
from pytorch_lightning.utilities.rank_zero import rank_zero_only

from src.utils.callbacks import (
    VAELoggingCallback,
    VAECheckpointCallback,
    LRandSchedulerOverrideCallback,
    NaNLossCallback,
    ConfigSnapshotCallback,
)
from src.config.base import VAETrainingConfig, EarlyStoppingConfig, CheckpointConfig, TrainingConfig


def merge_configs(train_config: dict, model_config: dict) -> dict:
    """Merge training and model configs."""
    config = {}
    config.update(train_config)
    config["model"] = model_config.get("model", {})
    return config


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


def build_trainer_kwargs(config: VAETrainingConfig) -> Dict[str, Any]:
    """Build Trainer kwargs with optional DDP/multi-GPU support."""
    distributed_config = config.distributed

    accelerator = distributed_config.accelerator
    devices: Union[str, int, List[int]] = distributed_config.devices
    strategy: Union[str, DDPStrategy] = distributed_config.strategy
    num_nodes = int(distributed_config.num_nodes)

    if isinstance(strategy, str) and strategy.lower() == "ddp":
        strategy = DDPStrategy(
            find_unused_parameters=distributed_config.find_unused_parameters,
            gradient_as_bucket_view=distributed_config.gradient_as_bucket_view,
        )

    trainer_kwargs = {
        "accelerator": accelerator,
        "devices": devices,
        "strategy": strategy,
        "num_nodes": num_nodes,
    }

    if distributed_config.sync_batchnorm:
        trainer_kwargs["sync_batchnorm"] = True

    return trainer_kwargs


def build_early_stopping_callback(early_stopping_config: EarlyStoppingConfig, checkpoint_config: CheckpointConfig) -> EarlyStopping | None:
    """Create an early stopping callback when the feature is enabled."""

    if not early_stopping_config.enabled:
        return None

    return EarlyStopping(
        monitor=early_stopping_config.monitor or checkpoint_config.monitor,
        mode=early_stopping_config.mode or checkpoint_config.mode,
        patience=max(0, early_stopping_config.patience),
        min_delta=early_stopping_config.min_delta,
        check_finite=early_stopping_config.check_finite,
        stopping_threshold=early_stopping_config.stopping_threshold,
        divergence_threshold=early_stopping_config.divergence_threshold,
        verbose=early_stopping_config.verbose,
    )


def build_callbacks(cfg: VAETrainingConfig) -> list:
    training_cfg = cfg.training
    path_cfg = cfg.paths
    checkpoint_cfg = cfg.checkpoint
    logging_cfg = cfg.logging
    early_stopping_cfg = cfg.early_stopping

    callbacks = [
        # NaN/Inf loss guard
        NaNLossCallback(),
        # Checkpoint callback
        VAECheckpointCallback(
            dirpath=path_cfg.checkpoint_dir,
            filename="vae-{epoch:02d}-{step:06d}-{val/rec_loss:.4f}",
            save_top_k=checkpoint_cfg.save_top_k,
            monitor=checkpoint_cfg.monitor,
            mode=checkpoint_cfg.mode,
            save_last=checkpoint_cfg.save_last,
            every_n_train_steps=checkpoint_cfg.save_every_n_steps,
            save_hf_format=True,
        ),
        # Learning rate monitor
        LearningRateMonitor(logging_interval="step"),
        # Unified VAE metrics/images logging
        VAELoggingCallback(
            num_val_images=logging_cfg.num_val_images,
            log_to_tensorboard=True,
            log_images_every_n_steps=logging_cfg.log_images_every_n_steps,
        ),
        # LR and Scheduler Override callback for resume
        LRandSchedulerOverrideCallback(
            override_lr_on_resume=training_cfg.override_lr_on_resume,
            reset_scheduler_on_resume=training_cfg.reset_scheduler_on_resume,
            gen_opt_config=training_cfg.generator_optimizer,
            disc_opt_config=training_cfg.discriminator_optimizer,
            verbose=True,
        ),
        # Config snapshot callback
        ConfigSnapshotCallback(cfg),
        # Progress bar
        RichProgressBar(),
    ]

    # Optional early stopping
    early_stopping_callback = build_early_stopping_callback(early_stopping_cfg, checkpoint_cfg)
    if early_stopping_cfg.enabled:
        callbacks.append(early_stopping_callback)

    return callbacks


def configure_cuda_precision(device_cfg: TrainingConfig) -> None:
    if not torch.cuda.is_available():
        return

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    torch.backends.cudnn.allow_tf32 = device_cfg.allow_tf32
    torch.set_float32_matmul_precision("high" if device_cfg.allow_tf32 else "highest")


def build_wandb_logger(cfg: VAETrainingConfig) -> WandbLogger:
    path_config = cfg.paths
    logging_config = cfg.logging

    return WandbLogger(
        project=logging_config.project,
        name=logging_config.name,
        save_dir=path_config.log_dir,
    )


def seed_everything(seed: int) -> None:
    pl.seed_everything(seed, workers=True)
