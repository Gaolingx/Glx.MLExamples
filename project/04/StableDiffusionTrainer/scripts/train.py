import argparse
from pathlib import Path
import os
import sys

import pytorch_lightning as pl

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.sd_datamodule import StableDiffusionDataModule
from src.lightningmodule.sd15_module import StableDiffusionLightningModule
from src.utils.config import load_json_config
from src.utils.training import (
    build_callbacks,
    build_tensorboard_logger,
    find_resume_checkpoint,
    seed_everything,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Stable Diffusion 1.5 with PyTorch Lightning.")
    parser.add_argument("--config", type=str, default="./configs/train_config.json", help="Path to JSON config file.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Auto-resume from latest checkpoint in checkpoint.dirpath.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_json_config(args.config)

    seed_everything(int(cfg.get("seed", 42)))

    model = StableDiffusionLightningModule(cfg)
    datamodule = StableDiffusionDataModule(
        dataset_cfg=cfg["dataset"],
        tokenizer=model.tokenizer,
        train_batch_size=int(cfg["training"].get("batch_size", 1)),
        seed=int(cfg.get("seed", 42)),
    )

    logger = build_tensorboard_logger(cfg.get("logging", {}))
    callbacks = build_callbacks(cfg)

    output_dir = Path(cfg.get("output_dir", "outputs/sd15-lightning"))
    output_dir.mkdir(parents=True, exist_ok=True)

    precision = cfg["training"].get("precision", "16-mixed")

    trainer = pl.Trainer(
        default_root_dir=str(output_dir),
        logger=logger,
        callbacks=callbacks,
        accelerator="auto",
        devices="auto",
        strategy="auto",
        max_epochs=int(cfg["training"].get("max_epochs", 1)),
        max_steps=int(cfg["training"].get("max_steps", -1)),
        accumulate_grad_batches=int(cfg["training"].get("accumulate_grad_batches", 1)),
        gradient_clip_val=float(cfg["training"].get("max_grad_norm", 1.0)),
        precision=precision,
        log_every_n_steps=1,
        num_sanity_val_steps=0,
    )

    ckpt_path = None
    if args.resume:
        ckpt_path = find_resume_checkpoint(cfg.get("checkpoint", {}))

    # Train
    print("=" * 60)
    print("Starting sd15 Training")
    print(f"  - Gradient accumulation: {int(cfg["training"].get("accumulate_grad_batches", 1))} steps")
    print(f"Output directory: {output_dir}")
    print(f"Log directory: {cfg["training"].get("tensorboard_save_dir", "outputs/sd15-lightning/tb_logs")}")
    print(f"Checkpoint directory: {cfg["checkpoint"].get("dirpath", "outputs/sd15-lightning/checkpoints")}")
    print(f"TensorBoard: tensorboard --logdir {cfg["training"].get("tensorboard_save_dir", "outputs/sd15-lightning/tb_logs")}")
    print("=" * 60)

    trainer.fit(model=model, datamodule=datamodule, ckpt_path=ckpt_path)

    # Save final model
    if trainer.is_global_zero:
        final_model_path = os.path.join(output_dir, "final_model")
        trainer.lightning_module.save_pretrained(final_model_path)
        print(f"Final model saved to: {final_model_path}")


if __name__ == "__main__":
    main()
