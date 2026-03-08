#!/usr/bin/env python
"""
Training script for AutoencoderKL with PyTorch Lightning.
Supports TensorBoard logging, automatic checkpointing, and resuming from checkpoints.

Features:
- Alternating training between VAE and Discriminator (official diffusers style)
- Manual gradient accumulation for proper alternating training
- Supports mixed precision training
- LR and scheduler override on resume

Usage:
    python scripts/train.py --config configs/train_config.json
    python scripts/train.py --config configs/train_config.json --resume_from_checkpoint path/to/checkpoint.ckpt
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    EarlyStopping,
    RichProgressBar,
)
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.utilities.rank_zero import rank_zero_only

from src.lightning.vae_module import VAELightningModule
from src.data.dataset import VAEDataModule
from src.utils.callbacks import (
    VAELoggingCallback,
    VAECheckpointCallback,
    LRandSchedulerOverrideCallback,
    NaNLossCallback,
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train AutoencoderKL with PyTorch Lightning"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train_config.json",
        help="Path to training config file",
    )
    parser.add_argument(
        "--model_config",
        type=str,
        default="configs/model_config.json",
        help="Path to model config file",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (overrides config)",
    )
    parser.add_argument(
        "--gpus",
        type=int,
        default=None,
        help="Number of GPUs to use",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default=None,
        choices=["16", "32", "bf16", "16-mixed", "bf16-mixed"],
        help="Training precision",
    )
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load configuration from JSON file."""
    with open(config_path, "r") as f:
        return json.load(f)


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


def resolve_parallel_devices(
        accelerator: str, devices: Union[str, int, List[int]]
) -> Optional[List[torch.device]]:
    """Resolve explicit parallel devices for DDPStrategy when GPU ids are provided."""
    if accelerator != "gpu":
        return None

    if isinstance(devices, list):
        gpu_ids = devices
    elif isinstance(devices, int):
        gpu_ids = list(range(devices))
    else:
        return None

    return [torch.device(f"cuda:{i}") for i in gpu_ids]


def build_trainer_kwargs(config: dict, args: argparse.Namespace) -> Dict[str, Any]:
    """Build Trainer kwargs with optional DDP/multi-GPU support."""
    train_config = config.get("training", {})
    distributed_config = config.get("distributed", {})

    accelerator = distributed_config.get("accelerator", "auto")
    devices: Union[str, int, List[int]] = distributed_config.get("devices", "auto")
    strategy: Union[str, DDPStrategy] = distributed_config.get("strategy", "auto")

    if args.gpus is not None:
        accelerator = "gpu"
        devices = args.gpus
    elif accelerator == "gpu" and devices == "auto" and torch.cuda.is_available():
        devices = torch.cuda.device_count()

    if isinstance(strategy, str) and strategy.lower() == "ddp":
        parallel_devices = resolve_parallel_devices(accelerator, devices)
        strategy = DDPStrategy(
            parallel_devices=parallel_devices,
            find_unused_parameters=distributed_config.get("find_unused_parameters", False),
            gradient_as_bucket_view=distributed_config.get("gradient_as_bucket_view", True),
        )

    trainer_kwargs = {
        "accelerator": accelerator,
        "devices": devices,
        "strategy": strategy,
    }

    if distributed_config.get("sync_batchnorm", False):
        trainer_kwargs["sync_batchnorm"] = True

    return trainer_kwargs


@rank_zero_only
def rank_zero_print(message: str) -> None:
    """Print only on global rank zero to avoid duplicated logs in DDP."""
    print(message)


@rank_zero_only
def save_runtime_config(config: dict, config_save_path: str) -> None:
    """Persist merged runtime config only once in distributed training."""
    with open(config_save_path, "w") as f:
        json.dump(config, f, indent=2)


def main():
    """Main training function."""
    args = parse_args()

    # Load configs
    train_config = load_config(args.config)
    model_config = load_config(args.model_config)
    config = merge_configs(train_config, model_config)

    # Override with command line arguments
    if args.seed is not None:
        config["training"]["seed"] = args.seed
    if args.precision is not None:
        config["training"]["precision"] = args.precision

    # Set seed
    seed = config.get("training", {}).get("seed", 42)
    pl.seed_everything(seed, workers=True)

    # Get paths
    paths = config.get("paths", {})
    output_dir = paths.get("output_dir", "./outputs")
    log_dir = paths.get("log_dir", "./logs")
    checkpoint_dir = paths.get("checkpoint_dir", "./checkpoints")

    # Create directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Save config only once to avoid duplicated writes in DDP
    config_save_path = os.path.join(output_dir, "config.json")
    save_runtime_config(config, config_save_path)

    # Create data module
    data_module = VAEDataModule(config)

    # Create model
    model = VAELightningModule(config)

    # Setup callbacks
    train_config_section = config.get("training", {})
    checkpoint_config = config.get("checkpoint", {})
    logging_config = config.get("logging", {})

    gen_opt_cfg = train_config_section.get("generator_optimizer", {})
    disc_opt_cfg = train_config_section.get("discriminator_optimizer", {})

    # Setup logger
    logger = TensorBoardLogger(
        save_dir=log_dir,
        name=logging_config.get("name", "vae_training"),
        default_hp_metric=False,
    )

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if train_config_section.get("allow_tf32", False):
        torch.backends.cuda.matmul.allow_tf32 = True

    callbacks = [
        # NaN/Inf loss guard
        NaNLossCallback(),
        # Checkpoint callback
        VAECheckpointCallback(
            dirpath=checkpoint_dir,
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
            override_lr_on_resume=train_config_section.get("override_lr_on_resume", False),
            reset_scheduler_on_resume=train_config_section.get("reset_scheduler_on_resume", False),
            gen_opt_config=gen_opt_cfg,
            disc_opt_config=disc_opt_cfg,
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

    # Create trainer
    # NOTE: accumulate_grad_batches is set to 1 because we handle gradient accumulation
    # manually in VAELightningModule.training_step() for proper alternating training
    # between VAE and Discriminator (following official diffusers implementation)
    trainer_kwargs = build_trainer_kwargs(config, args)
    trainer = pl.Trainer(
        default_root_dir=output_dir,
        max_epochs=train_config_section.get("num_epochs", 100),
        max_steps=train_config_section.get("max_train_steps", -1),
        precision=train_config_section.get("precision", "16-mixed"),
        accumulate_grad_batches=1,  # Manual accumulation in training_step
        log_every_n_steps=logging_config.get("log_every_n_steps", 50),
        val_check_interval=logging_config.get("val_check_interval", 500),
        logger=logger,
        callbacks=callbacks,
        enable_checkpointing=True,
        deterministic=True,
        **trainer_kwargs,
    )

    ckpt_path = None
    if args.resume_from_checkpoint:
        ckpt_path = find_resume_checkpoint(args.resume_from_checkpoint, checkpoint_dir)

    # Train
    distributed_config = config.get("distributed", {})
    requested_devices = args.gpus if args.gpus is not None else distributed_config.get("devices", "auto")
    requested_strategy = distributed_config.get("strategy", "auto")

    rank_zero_print("=" * 60)
    rank_zero_print("Starting VAE Training")
    rank_zero_print(f"  - Gradient accumulation: {train_config_section.get('accumulate_grad_batches', 1)} steps")
    rank_zero_print(f"  - Override LR on resume: {train_config_section.get('override_lr_on_resume', True)}")
    rank_zero_print(f"  - Reset scheduler on resume: {train_config_section.get('reset_scheduler_on_resume', False)}")
    rank_zero_print(f"  - Accelerator: {distributed_config.get('accelerator', 'auto')}")
    rank_zero_print(f"  - Devices: {requested_devices}")
    rank_zero_print(f"  - Strategy: {requested_strategy}")
    rank_zero_print(f"Output directory: {output_dir}")
    rank_zero_print(f"Log directory: {log_dir}")
    rank_zero_print(f"Checkpoint directory: {checkpoint_dir}")
    rank_zero_print(f"TensorBoard: tensorboard --logdir {log_dir}")
    rank_zero_print("=" * 60)

    trainer.fit(model, datamodule=data_module, ckpt_path=ckpt_path)

    # Save final model
    if trainer.is_global_zero:
        final_model_path = os.path.join(output_dir, "final_model")
        trainer.lightning_module.save_pretrained(final_model_path)
        print(f"Final model saved to: {final_model_path}")


if __name__ == "__main__":
    main()
