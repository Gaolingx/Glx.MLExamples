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
import sys
import os
from pathlib import Path

# Add project root to path
__package__ = "scripts"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import pytorch_lightning as pl

from src.lightning.vae_module import VAELightningModule
from src.data.dataset import VAEDataModule

from src.config.loader import load_vae_training_config
from src.utils.training import (
    seed_everything,
    build_callbacks,
    build_wandb_logger,
    build_trainer_kwargs,
    find_resume_checkpoint,
    configure_cuda_precision,
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train AutoencoderKL with PyTorch Lightning"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="./configs/train_config.json",
        help="Path to training config file",
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
        "--precision",
        type=str,
        default=None,
        choices=["16", "32", "bf16", "16-mixed", "bf16-mixed"],
        help="Training precision",
    )
    parser.add_argument(
        "--gpus",
        type=int,
        default=None,
        help="Number of GPUs to use",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory",
    )
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()

    # Load configs
    config = load_vae_training_config(args.config)

    # Override with command line arguments
    if args.seed is not None:
        config.training.seed = args.seed
    if args.precision is not None:
        config.training.precision = args.precision
    if args.gpus is not None:
        config.distributed.devices = args.gpus
    if args.output_dir is not None:
        config.paths.output_dir = args.output_dir

    training_config = config.training
    logging_config = config.logging
    path_config = config.paths

    # Set seed
    seed_everything(int(training_config.seed))

    # Get paths
    output_dir = Path(path_config.output_dir)

    # Create directories
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create data module
    data_module = VAEDataModule(config)

    # Create model
    model = VAELightningModule(config)

    # Setup callbacks
    callbacks = build_callbacks(config)

    # Setup logger
    logger = build_wandb_logger(config)

    configure_cuda_precision(config.training)

    # Create trainer
    # NOTE: accumulate_grad_batches is set to 1 because we handle gradient accumulation
    # manually in VAELightningModule.training_step() for proper alternating training
    # between VAE and Discriminator (following official diffusers implementation)
    trainer_kwargs = build_trainer_kwargs(config)
    trainer = pl.Trainer(
        default_root_dir=output_dir,
        max_epochs=training_config.num_epochs,
        max_steps=training_config.max_train_steps,
        precision=training_config.precision,
        accumulate_grad_batches=1,  # Manual accumulation in training_step
        log_every_n_steps=logging_config.log_every_n_steps,
        val_check_interval=logging_config.val_check_interval,
        logger=logger,
        callbacks=callbacks,
        enable_checkpointing=True,
        deterministic=True,
        **trainer_kwargs,
    )

    ckpt_path = None
    if args.resume_from_checkpoint:
        ckpt_path = find_resume_checkpoint(
            args.resume_from_checkpoint,
            path_config.checkpoint_dir,
        )

    trainer.fit(model, datamodule=data_module, ckpt_path=ckpt_path)


if __name__ == "__main__":
    main()
