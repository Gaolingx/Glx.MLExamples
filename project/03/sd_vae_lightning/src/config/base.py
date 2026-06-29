"""
Central configuration dataclasses for the VAE training pipeline.
Supports loading from JSON using dacite and provides type-safe access.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Union


# ---------------------------
# DataLoader configuration
# ---------------------------
@dataclass
class DataConfig:
    dataset_name: Optional[str] = None
    cache_dir: Optional[str] = None
    dataset_config: Optional[str] = None
    train_data_dir: Optional[str] = None
    val_data_dir: Optional[str] = None
    image_column: str = "image"
    resolution: int = 512
    center_crop: bool = True
    random_flip: bool = True

# ---------------------------
# Optimizer configuration
# ---------------------------
@dataclass
class OptimizerConfig:
    type: str = "adamw"
    learning_rate: float = 1e-4
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    weight_decay: float = 0.0
    momentum: float = 0.0
    use_8bit_adam: bool = False

    lr_scheduler: str = "constant"
    lr_warmup_steps: int = 0
    lr_num_cycles: float = 1.0
    lr_power: float = 1.0
    lr_step_rules: Optional[str] = None


# ---------------------------
# Model configuration
# ---------------------------
@dataclass
class ModelConfig:
    pretrained_model_name_or_path: Optional[str] = None
    model_config_name_or_path: Optional[str] = None
    vae_config: Optional[Dict[str, Any]] = None
    decoder_only: bool = False
    gradient_checkpointing: bool = False


# ---------------------------
# Loss configuration
# ---------------------------
@dataclass
class LossConfig:
    disc_weight: float = 1.0
    disc_start_step: int = 50001
    perceptual_weight: float = 0.5
    rec_loss_type: str = "l2"  # "l1" or "l2"
    kl_weight: float = 1e-6
    disc_loss_type: str = "hinge"  # "hinge" or "vanilla"
    disc_factor: float = 1.0
    use_adaptive_disc_weight: bool = True
    adaptive_weight_max: float = 10.0


# ---------------------------
# Discriminator configuration
# ---------------------------
@dataclass
class DiscriminatorConfig:
    input_nc: int = 3
    ndf: int = 64
    n_layers: int = 3
    norm_type: str = "spectral_group"
    num_groups: int = 32


# ---------------------------
# Training configuration
# ---------------------------
@dataclass
class TrainingConfig:
    seed: int = 42
    num_epochs: int = 100
    max_train_steps: int = -1
    precision: str = "16-mixed"
    allow_tf32: bool = False
    batch_size: int = 4
    num_workers: int = 4

    generator_optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    discriminator_optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)

    accumulate_grad_batches: int = 1
    gradient_clip_val: float = 1.0

    use_ema: bool = True
    ema_decay: float = 0.9999
    ema_update_interval: int = 1

    override_lr_on_resume: bool = False
    reset_scheduler_on_resume: bool = False


# ---------------------------
# Compile configuration (torch.compile)
# ---------------------------
@dataclass
class CompileConfig:
    enabled: bool = False
    kwargs: Dict[str, Any] = field(default_factory=dict)


# ---------------------------
# Paths configuration
# ---------------------------
@dataclass
class PathsConfig:
    output_dir: str = "./outputs"
    checkpoint_dir: str = "./checkpoints"
    log_dir: str = "./logs"


# ---------------------------
# Logging configuration
# ---------------------------
@dataclass
class LoggingConfig:
    project: str = "default"
    name: str = "vae_training"
    num_val_images: int = 4
    log_images_every_n_steps: int = 500
    log_every_n_steps: int = 50
    val_check_interval: Union[int, float] = 500


# ---------------------------
# Checkpointing configuration
# ---------------------------
@dataclass
class CheckpointConfig:
    save_top_k: int = 3
    monitor: str = "val/rec_loss"
    mode: str = "min"
    save_last: bool = True
    save_every_n_steps: int = 1000


# ---------------------------
# Distributed training configuration
# ---------------------------
@dataclass
class DistributedConfig:
    accelerator: str = "auto"
    devices: Union[str, int, List[int]] = "auto"
    strategy: str = "auto"
    num_nodes: int = 1
    find_unused_parameters: bool = False
    gradient_as_bucket_view: bool = True
    sync_batchnorm: bool = False


# ---------------------------
# Early stopping configuration
# ---------------------------
@dataclass
class EarlyStoppingConfig:
    enabled: bool = False
    monitor: Optional[str] = None  # if None, uses checkpoint monitor
    mode: Optional[str] = None     # if None, uses checkpoint mode
    patience: int = 3
    min_delta: float = 0.0
    check_finite: bool = True
    stopping_threshold: Optional[float] = None
    divergence_threshold: Optional[float] = None
    verbose: bool = False


# ---------------------------
# Root configuration
# ---------------------------
@dataclass
class VAETrainingConfig:
    vae_config_path: str = "./configs/model_config.json"

    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    discriminator: DiscriminatorConfig = field(default_factory=DiscriminatorConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    compile: CompileConfig = field(default_factory=CompileConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    distributed: DistributedConfig = field(default_factory=DistributedConfig)
    early_stopping: EarlyStoppingConfig = field(default_factory=EarlyStoppingConfig)
