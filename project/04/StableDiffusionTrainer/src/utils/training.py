import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.loggers import TensorBoardLogger

from src.utils.callbacks import (
    CheckpointCallback,
    NaNLossCallback,
    LRandSchedulerOverrideCallback,
    GradParamNormCallback,
    LoggingCallback,
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
        filename="sd15-{epoch:02d}-{step:06d}-{train/loss:.4f}",
        monitor=checkpoint_cfg.get("monitor", "train/loss"),
        mode=checkpoint_cfg.get("mode", "min"),
        save_top_k=int(checkpoint_cfg.get("save_top_k", 3)),
        save_last=bool(checkpoint_cfg.get("save_last", True)),
        every_n_train_steps=int(checkpoint_cfg.get("every_n_train_steps", 100)),
        save_hf_format=True,
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
