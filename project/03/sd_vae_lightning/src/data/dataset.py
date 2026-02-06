"""
Dataset and DataModule for VAE training.
Supports loading from HuggingFace Hub or local directories.
"""

from typing import Optional, Dict, Any, Callable, List
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from torchvision import transforms
from PIL import Image

try:
    from datasets import load_dataset

    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False


class VAEDataset(Dataset):
    """
    Dataset for VAE training.

    Args:
        data: List of image paths or HuggingFace dataset.
        transform: Image transformation pipeline.
        image_column: Column name for images in dataset.
    """

    def __init__(
            self,
            data: Any,
            transform: Optional[Callable] = None,
            image_column: str = "image",
    ):
        self.data = data
        self.transform = transform
        self.image_column = image_column

    def __len__(self) -> int:
        """Return dataset length."""
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get item by index.

        Args:
            idx: Sample index.

        Returns:
            Dictionary containing 'pixel_values' tensor.
        """
        item = self.data[idx]

        # Handle different data formats
        if isinstance(item, dict):
            image = item[self.image_column]
        elif isinstance(item, (str, Path)):
            image = Image.open(item).convert("RGB")
        else:
            image = item

        # Ensure PIL Image
        if not isinstance(image, Image.Image):
            if hasattr(image, "convert"):
                image = image.convert("RGB")
            else:
                image = Image.fromarray(image).convert("RGB")

        # Apply transforms
        if self.transform is not None:
            image = self.transform(image)

        return {"pixel_values": image}


class VAEDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for VAE training.

    Supports:
    - HuggingFace datasets
    - Local image directories
    - Custom dataset splits

    Args:
        config: Configuration dictionary.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.data_config = config.get("data", {})
        self.train_config = config.get("training", {})

        self.dataset_name = self.data_config.get("dataset_name")
        self.dataset_config = self.data_config.get("dataset_config")
        self.train_data_dir = self.data_config.get("train_data_dir")
        self.val_data_dir = self.data_config.get("val_data_dir")
        self.image_column = self.data_config.get("image_column", "image")
        self.resolution = self.data_config.get("resolution", 512)
        self.center_crop = self.data_config.get("center_crop", True)
        self.random_flip = self.data_config.get("random_flip", True)

        self.batch_size = self.train_config.get("batch_size", 4)
        self.num_workers = self.train_config.get("num_workers", 4)

        self.train_dataset = None
        self.val_dataset = None

    def prepare_data(self) -> None:
        """Download data if needed. Called on single GPU."""
        if self.dataset_name and HAS_DATASETS:
            # This will download the dataset if not cached
            load_dataset(
                self.dataset_name,
                self.dataset_config,
                split="train",
            )

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Setup datasets. Called on every GPU.

        Args:
            stage: 'fit', 'validate', 'test', or 'predict'.
        """
        # Build transforms
        train_transform = self._build_transform(augment=True)
        val_transform = self._build_transform(augment=False)

        if stage == "fit" or stage is None:
            # Load training data
            if self.dataset_name and HAS_DATASETS:
                train_data = load_dataset(
                    self.dataset_name,
                    self.dataset_config,
                    split="train",
                )
                # Check if validation split exists
                try:
                    val_data = load_dataset(
                        self.dataset_name,
                        self.dataset_config,
                        split="validation",
                    )
                except ValueError:
                    # Split training data for validation
                    split = train_data.train_test_split(test_size=0.1, seed=42)
                    train_data = split["train"]
                    val_data = split["test"]
            else:
                # Load from local directory
                train_data = self._load_local_images(self.train_data_dir)
                val_data = self._load_local_images(
                    self.val_data_dir or self.train_data_dir
                )

            self.train_dataset = VAEDataset(
                data=train_data,
                transform=train_transform,
                image_column=self.image_column,
            )
            self.val_dataset = VAEDataset(
                data=val_data,
                transform=val_transform,
                image_column=self.image_column,
            )

        if stage == "validate" or stage is None:
            if self.val_dataset is None:
                if self.dataset_name and HAS_DATASETS:
                    try:
                        val_data = load_dataset(
                            self.dataset_name,
                            self.dataset_config,
                            split="validation",
                        )
                    except ValueError:
                        val_data = load_dataset(
                            self.dataset_name,
                            self.dataset_config,
                            split="train[:10%]",
                        )
                else:
                    val_data = self._load_local_images(
                        self.val_data_dir or self.train_data_dir
                    )

                self.val_dataset = VAEDataset(
                    data=val_data,
                    transform=val_transform,
                    image_column=self.image_column,
                )

    def _build_transform(self, augment: bool = False) -> transforms.Compose:
        """
        Build image transformation pipeline.

        Args:
            augment: Whether to apply data augmentation.

        Returns:
            Composed transform.
        """
        transform_list = []

        # Resize
        transform_list.append(
            transforms.Resize(
                self.resolution, interpolation=transforms.InterpolationMode.BILINEAR
            )
        )

        # Crop
        if self.center_crop:
            transform_list.append(transforms.CenterCrop(self.resolution))
        else:
            if augment:
                transform_list.append(transforms.RandomCrop(self.resolution))
            else:
                transform_list.append(transforms.CenterCrop(self.resolution))

        # Horizontal flip (training only)
        if augment and self.random_flip:
            transform_list.append(transforms.RandomHorizontalFlip(p=0.5))

        # To tensor and normalize to [-1, 1]
        transform_list.append(transforms.ToTensor())
        transform_list.append(transforms.Normalize([0.5], [0.5]))

        return transforms.Compose(transform_list)

    def _load_local_images(self, data_dir: str) -> List[str]:
        """
        Load image paths from local directory.

        Args:
            data_dir: Path to image directory.

        Returns:
            List of image paths.
        """
        if data_dir is None:
            return []

        data_path = Path(data_dir)
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        image_paths = []

        for ext in image_extensions:
            image_paths.extend(data_path.rglob(f"*{ext}"))
            image_paths.extend(data_path.rglob(f"*{ext.upper()}"))

        return sorted([str(p) for p in image_paths])

    def train_dataloader(self) -> DataLoader:
        """Return training dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        """Return validation dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
        )
