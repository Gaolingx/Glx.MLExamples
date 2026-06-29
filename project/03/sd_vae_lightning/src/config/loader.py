"""
Configuration loading utilities.
Safe to import from other modules without causing circular imports.
"""

import dacite

from src.config.base import VAETrainingConfig
from src.utils.config import load_json_config


def load_vae_training_config(config_path: str) -> VAETrainingConfig:
    """Load VAE training config using dacite if available."""

    train_config = load_json_config(config_path)
    vae_config_path = train_config.get("vae_config_path", "./configs/model_config.json")
    model_config = load_json_config(vae_config_path)
    
    config = {}
    config.update(train_config)
    config["model"] = model_config.get("model", {})
    
    return dacite.from_dict(
        data_class=VAETrainingConfig,
        data=config,
        config=dacite.Config(strict=False),
    )
