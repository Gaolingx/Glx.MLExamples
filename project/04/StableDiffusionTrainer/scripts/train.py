import argparse
from pathlib import Path
import sys

import torch
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
    build_trainer_kwargs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Stable Diffusion 1.5 with PyTorch Lightning.")
    parser.add_argument(
        "--config",
        type=str,
        default="./configs/train_config.json",
        help="Path to JSON config file.",
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
        "--lora_init_path",
        type=str,
        default=None,
        help="Optional LoRA init path. Supports adapter dir or Lightning ckpt exported with hf_checkpoint/unet_lora.",
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Upload final exported model artifacts to the Hugging Face Hub.",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="Optional Hub repo id used when --push_to_hub is enabled.",
    )
    parser.add_argument(
        "--hub_token",
        type=str,
        default=None,
        help="Optional Hugging Face token used for uploads.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_json_config(args.config)

    training_cfg = cfg.get("training", {})
    logging_cfg = cfg.get("logging", {})
    checkpoint_cfg = cfg.get("checkpoint", {})

    # Override with command line arguments
    if args.seed is not None:
        cfg["training"]["seed"] = args.seed
    if args.precision is not None:
        cfg["training"]["precision"] = args.precision
    if args.gpus is not None:
        cfg["distributed"]["devices"] = args.gpus
    if args.lora_init_path is not None:
        cfg.setdefault("lora", {})["enabled"] = True
        cfg["lora"]["init_path"] = args.lora_init_path
    if args.push_to_hub:
        cfg.setdefault("hub", {})["push_to_hub"] = True
    if args.hub_model_id is not None:
        cfg.setdefault("hub", {})["model_id"] = args.hub_model_id
    if args.hub_token is not None:
        cfg.setdefault("hub", {})["token"] = args.hub_token

    seed_everything(int(training_cfg.get("seed", 42)))

    allow_tf32 = bool(training_cfg.get("allow_tf32", False))
    torch.backends.cudnn.allow_tf32 = allow_tf32
    torch.set_float32_matmul_precision("high" if allow_tf32 else "highest")

    model = StableDiffusionLightningModule(cfg)
    datamodule = StableDiffusionDataModule(
        dataset_cfg=cfg["dataset"],
        tokenizer=model.tokenizer,
        train_batch_size=int(training_cfg.get("batch_size", 1)),
        seed=int(training_cfg.get("seed", 42)),
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
        **trainer_distributed_kwargs,
    )

    ckpt_path = None
    if args.resume_from_checkpoint:
        ckpt_dir = checkpoint_cfg.get("dirpath", "outputs/checkpoints")
        ckpt_path = find_resume_checkpoint(args.resume_from_checkpoint, ckpt_dir)

    trainer.fit(model=model, datamodule=datamodule, ckpt_path=ckpt_path)


if __name__ == "__main__":
    main()
