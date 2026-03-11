import argparse
from pathlib import Path
import os
import sys
from typing import Any, Dict, List, Optional, Union

import torch
import pytorch_lightning as pl

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from pytorch_lightning.strategies import DDPStrategy
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


def build_trainer_kwargs(config: dict) -> Dict[str, Any]:
    """Build Trainer kwargs with optional DDP/multi-GPU support."""
    distributed_config = config.get("distributed", {})

    accelerator = distributed_config.get("accelerator", "auto")
    devices: Union[str, int, List[int]] = distributed_config.get("devices", "auto")
    strategy: Union[str, DDPStrategy] = distributed_config.get("strategy", "auto")
    num_nodes = int(distributed_config.get("num_nodes", 1))

    if accelerator == "gpu" and devices == "auto" and torch.cuda.is_available():
        devices = torch.cuda.device_count()

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


def main() -> None:
    args = parse_args()
    cfg = load_json_config(args.config)

    training_cfg = cfg.get("training", {})
    logging_cfg = cfg.get("logging", {})
    checkpoint_cfg = cfg.get("checkpoint", {})

    seed_everything(int(cfg.get("seed", 42)))

    allow_tf32 = bool(training_cfg.get("allow_tf32", training_cfg.get("tf32", False)))
    torch.backends.cudnn.allow_tf32 = allow_tf32
    torch.set_float32_matmul_precision("high" if allow_tf32 else "highest")

    model = StableDiffusionLightningModule(cfg)
    datamodule = StableDiffusionDataModule(
        dataset_cfg=cfg["dataset"],
        tokenizer=model.tokenizer,
        train_batch_size=int(training_cfg.get("batch_size", 1)),
        seed=int(cfg.get("seed", 42)),
    )

    logger = build_tensorboard_logger(logging_cfg)
    callbacks = build_callbacks(cfg)

    output_dir = Path(cfg.get("output_dir", "outputs/sd15-lightning"))
    output_dir.mkdir(parents=True, exist_ok=True)

    precision = str(training_cfg.get("precision", "16-mixed"))
    trainer_distributed_kwargs = build_trainer_kwargs(cfg)

    trainer = pl.Trainer(
        default_root_dir=str(output_dir),
        logger=logger,
        callbacks=callbacks,
        max_epochs=int(training_cfg.get("max_epochs", 1)),
        max_steps=int(training_cfg.get("max_steps", -1)),
        accumulate_grad_batches=int(training_cfg.get("accumulate_grad_batches", 1)),
        gradient_clip_val=float(training_cfg.get("max_grad_norm", 1.0)),
        log_every_n_steps=int(logging_cfg.get("log_every_n_steps", 1)),
        precision=precision,
        num_sanity_val_steps=0,
        **trainer_distributed_kwargs,
    )

    ckpt_path = None
    if args.resume:
        ckpt_path = find_resume_checkpoint(checkpoint_cfg)

    trainer.fit(model=model, datamodule=datamodule, ckpt_path=ckpt_path)

    # Save final model
    if trainer.is_global_zero:
        final_model_path = os.path.join(output_dir, "final_model")
        trainer.lightning_module.save_pretrained(final_model_path)
        print(f"Final model saved to: {final_model_path}")


if __name__ == "__main__":
    main()
