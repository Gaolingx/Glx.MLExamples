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

from src.lightning.vae_module import VAELightningModule
from src.data.dataset import VAEDataModule
from src.utils.callbacks import (
    ImageLoggerCallback,
    VAECheckpointCallback,
    GradientNormLogger,
    LRandSchedulerOverrideCallback,
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
    if "pretrained_model_name_or_path" in model_config:
        config["model"]["pretrained_model_name_or_path"] = model_config[
            "pretrained_model_name_or_path"
        ]
    return config


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

    # Save config
    config_save_path = os.path.join(output_dir, "config.json")
    with open(config_save_path, "w") as f:
        json.dump(config, f, indent=2)

    # Create data module
    data_module = VAEDataModule(config)

    # Create model
    model = VAELightningModule(config)

    # Setup logger
    logger = TensorBoardLogger(
        save_dir=log_dir,
        name="vae_training",
        default_hp_metric=False,
    )

    # Setup callbacks
    train_config_section = config.get("training", {})
    checkpoint_config = config.get("checkpoint", {})
    logging_config = config.get("logging", {})

    callbacks = [
        # Checkpoint callback
        VAECheckpointCallback(
            dirpath=checkpoint_dir,
            filename="vae-{epoch:02d}-{step:06d}-{val/rec_loss:.4f}",
            save_top_k=checkpoint_config.get("save_top_k", 3),
            monitor=checkpoint_config.get("monitor", "val/rec_loss"),
            mode=checkpoint_config.get("mode", "min"),
            save_last=True,
            every_n_train_steps=checkpoint_config.get("save_every_n_steps"),
            save_hf_format=True,
        ),
        # Learning rate monitor
        LearningRateMonitor(logging_interval="step"),
        # Image logger
        ImageLoggerCallback(
            log_every_n_steps=logging_config.get("log_images_every_n_steps", 500),
            num_images=logging_config.get("num_val_images", 4),
            log_to_tensorboard=True,
        ),
        # Gradient norm logger
        GradientNormLogger(log_every_n_steps=100),
        # LR and Scheduler Override callback for resume
        LRandSchedulerOverrideCallback(
            override_lr_on_resume=train_config_section.get("override_lr_on_resume", True),
            reset_scheduler_on_resume=train_config_section.get("reset_scheduler_on_resume", False),
            vae_lr=train_config_section.get("learning_rate"),
            disc_lr=train_config_section.get("disc_learning_rate"),
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

    # Determine accelerator and devices
    if torch.cuda.is_available():
        accelerator = "gpu"
        devices = args.gpus if args.gpus else "auto"
    elif torch.backends.mps.is_available():
        accelerator = "mps"
        devices = 1
    else:
        accelerator = "cpu"
        devices = "auto"

    # Create trainer
    # NOTE: accumulate_grad_batches is set to 1 because we handle gradient accumulation
    # manually in VAELightningModule.training_step() for proper alternating training
    # between VAE and Discriminator (following official diffusers implementation)
    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        max_epochs=train_config_section.get("num_epochs", 100),
        precision=train_config_section.get("precision", "16-mixed"),
        accumulate_grad_batches=1,  # Manual accumulation in training_step
        # gradient_clip_val handled manually in training_step
        log_every_n_steps=logging_config.get("log_every_n_steps", 50),
        val_check_interval=logging_config.get("val_check_interval", 500),
        logger=logger,
        callbacks=callbacks,
        enable_checkpointing=True,
        deterministic=True,
    )

    # Handle checkpoint resuming
    ckpt_path = args.resume_from_checkpoint
    if ckpt_path == "latest":
        # Find latest checkpoint
        ckpt_files = list(Path(checkpoint_dir).glob("*.ckpt"))
        if ckpt_files:
            ckpt_path = str(max(ckpt_files, key=os.path.getctime))
            print(f"Resuming from latest checkpoint: {ckpt_path}")
        else:
            ckpt_path = None
            print("No checkpoints found, starting from scratch")

    # Train
    print("=" * 60)
    print("Starting VAE Training")
    print(f"  - Alternating training: VAE (even steps) / Disc (odd steps)")
    print(f"  - Gradient accumulation: {train_config_section.get('accumulate_grad_batches', 1)} steps")
    print(f"  - Override LR on resume: {train_config_section.get('override_lr_on_resume', True)}")
    print(f"  - Reset scheduler on resume: {train_config_section.get('reset_scheduler_on_resume', False)}")
    print(f"Output directory: {output_dir}")
    print(f"Log directory: {log_dir}")
    print(f"Checkpoint directory: {checkpoint_dir}")
    print(f"TensorBoard: tensorboard --logdir {log_dir}")
    print("=" * 60)

    trainer.fit(model, datamodule=data_module, ckpt_path=ckpt_path)

    # Save final model
    if trainer.is_global_zero:
        final_model_path = os.path.join(output_dir, "final_model")
        trainer.lightning_module.save_pretrained(final_model_path)
        print(f"Final model saved to: {final_model_path}")


if __name__ == "__main__":
    main()
