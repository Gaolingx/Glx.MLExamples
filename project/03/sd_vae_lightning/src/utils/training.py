import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
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
)


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


def build_callbacks(cfg: Dict[str, Any]) -> list:
    training_config = cfg.get("training", {})
    path_config = cfg.get("paths", {})
    checkpoint_config = cfg.get("checkpoint", {})
    logging_config = cfg.get("logging", {})

    callbacks = [
        # NaN/Inf loss guard
        NaNLossCallback(),
        # Checkpoint callback
        VAECheckpointCallback(
            dirpath=path_config.get("checkpoint_dir", "./checkpoints"),
            filename="vae-{epoch:02d}-{step:06d}-{val/rec_loss:.4f}",
            save_top_k=checkpoint_config.get("save_top_k", 3),
            monitor=checkpoint_config.get("monitor", "val/rec_loss"),
            mode=checkpoint_config.get("mode", "min"),
            save_last=True,
            every_n_train_steps=checkpoint_config.get("save_every_n_steps", 1000),
            save_hf_format=True,
        ),
        # Learning rate monitor
        LearningRateMonitor(logging_interval="step"),
        # Unified VAE metrics/images logging
        VAELoggingCallback(
            num_val_images=logging_config.get("num_val_images", 4),
            log_to_tensorboard=True,
            log_images_every_n_steps=logging_config.get("log_images_every_n_steps", 500),
        ),
        # LR and Scheduler Override callback for resume
        LRandSchedulerOverrideCallback(
            override_lr_on_resume=training_config.get("override_lr_on_resume", False),
            reset_scheduler_on_resume=training_config.get("reset_scheduler_on_resume", False),
            gen_opt_config=training_config.get("generator_optimizer", {}),
            disc_opt_config=training_config.get("discriminator_optimizer", {}),
            verbose=True,
        ),
        # Progress bar
        RichProgressBar(),
    ]

    # Optional early stopping
    if checkpoint_config.get("early_stopping", False):
        callbacks.append(
            EarlyStopping(
                monitor=checkpoint_config.get("monitor", "val/rec_loss"),
                patience=checkpoint_config.get("patience", 10),
                mode=checkpoint_config.get("mode", "min"),
            )
        )

    return callbacks


def build_tensorboard_logger(cfg: Dict[str, Any]) -> TensorBoardLogger:
    path_config = cfg.get("paths", {})
    logging_config = cfg.get("logging", {})

    return TensorBoardLogger(
        save_dir=path_config.get("log_dir", "./logs"),
        name=logging_config.get("name", "vae_training"),
        default_hp_metric=False,
    )


def seed_everything(seed: int) -> None:
    pl.seed_everything(seed, workers=True)
