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
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import pytorch_lightning as pl

from src.lightning.vae_module import VAELightningModule
from src.data.dataset import VAEDataModule

from src.utils.config import load_json_config
from src.utils.training import (
    merge_configs,
    seed_everything,
    save_runtime_config,
    build_callbacks,
    build_tensorboard_logger,
    build_trainer_kwargs,
    find_resume_checkpoint,
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
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()

    # Load configs
    train_config = load_json_config(args.config)
    model_config = load_json_config(train_config.get("vae_config_path", "./configs/model_config.json"))
    config = merge_configs(train_config, model_config)

    # Override with command line arguments
    if args.seed is not None:
        config["training"]["seed"] = args.seed
    if args.precision is not None:
        config["training"]["precision"] = args.precision
    if args.gpus is not None:
        config["distributed"]["devices"] = args.gpus

    training_config = config.get("training", {})
    logging_config = config.get("logging", {})
    path_config = config.get("paths", {})

    # Set seed
    seed_everything(int(training_config.get("seed", 42)))

    # Get paths
    output_dir = Path(path_config.get("output_dir", "./outputs"))

    # Create directories
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config only once to avoid duplicated writes in DDP
    config_save_path = Path(output_dir) / "config.json"
    save_runtime_config(config, config_save_path)

    # Create data module
    data_module = VAEDataModule(config)

    # Create model
    model = VAELightningModule(config)

    # Setup callbacks
    callbacks = build_callbacks(config)

    # Setup logger
    logger = build_tensorboard_logger(config)

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    torch.backends.cudnn.allow_tf32 = training_config.get("allow_tf32", False)
    torch.set_float32_matmul_precision("high" if training_config.get("allow_tf32", False) else "highest")

    # Create trainer
    # NOTE: accumulate_grad_batches is set to 1 because we handle gradient accumulation
    # manually in VAELightningModule.training_step() for proper alternating training
    # between VAE and Discriminator (following official diffusers implementation)
    trainer_kwargs = build_trainer_kwargs(config)
    trainer = pl.Trainer(
        default_root_dir=output_dir,
        max_epochs=training_config.get("num_epochs", 100),
        max_steps=training_config.get("max_train_steps", -1),
        precision=training_config.get("precision", "16-mixed"),
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
        ckpt_dir = path_config.get("checkpoint_dir", "./checkpoints")
        ckpt_path = find_resume_checkpoint(args.resume_from_checkpoint, ckpt_dir)

    trainer.fit(model, datamodule=data_module, ckpt_path=ckpt_path)

    # Save final model
    if trainer.is_global_zero:
        final_model_path = Path(output_dir) / "final_model"
        trainer.lightning_module.save_pretrained(final_model_path)
        print(f"Final model saved to: {final_model_path}")

if __name__ == "__main__":
    main()
