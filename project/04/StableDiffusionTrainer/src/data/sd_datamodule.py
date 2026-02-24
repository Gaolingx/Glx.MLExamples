from typing import Any, Dict, List, Optional

import pytorch_lightning as pl
import torch
from datasets import Dataset, load_dataset
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import CLIPTokenizer


class StableDiffusionDataModule(pl.LightningDataModule):
    """DataModule for image-caption training with Hugging Face datasets."""

    def __init__(
        self,
        dataset_cfg: Dict[str, Any],
        tokenizer: CLIPTokenizer,
        train_batch_size: int,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.dataset_cfg = dataset_cfg
        self.tokenizer = tokenizer
        self.train_batch_size = train_batch_size
        self.seed = seed
        self.image_column = self.dataset_cfg.get("image_column", "image")
        self.caption_column = self.dataset_cfg.get("caption_column", "text")

        self.dataset: Optional[Dataset] = None

        resolution = int(self.dataset_cfg.get("resolution", 512))
        center_crop = bool(self.dataset_cfg.get("center_crop", True))
        random_flip = bool(self.dataset_cfg.get("random_flip", True))

        crop_op = transforms.CenterCrop(resolution) if center_crop else transforms.RandomCrop(resolution)
        flip_op = transforms.RandomHorizontalFlip() if random_flip else transforms.Lambda(lambda x: x)

        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
                crop_op,
                flip_op,
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def prepare_data(self) -> None:
        """Download dataset metadata and cache shards from Hugging Face."""
        load_dataset(
            path=self.dataset_cfg["name"],
            split=self.dataset_cfg.get("split", "train"),
            cache_dir=self.dataset_cfg.get("cache_dir"),
        )

    def _transform_examples(self, examples: Dict[str, List[Any]]) -> Dict[str, Any]:
        images = examples[self.image_column]
        captions = examples[self.caption_column]
        if not isinstance(images, list):
            images = [images]
        if not isinstance(captions, list):
            captions = [captions]
        pixel_values = []
        for image in images:
            if not isinstance(image, Image.Image):
                image = Image.open(image).convert("RGB")
            else:
                image = image.convert("RGB")
            pixel_values.append(self.image_transforms(image))
        tokenized = self.tokenizer(
            captions,
            truncation=True,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )

        if len(pixel_values) == 1:
            return {
                "pixel_values": pixel_values[0],
                "input_ids": tokenized.input_ids[0],
            }

        return {
            "pixel_values": pixel_values,
            "input_ids": tokenized.input_ids,
        }

    def setup(self, stage: Optional[str] = None) -> None:
        if stage in (None, "fit"):
            dataset = load_dataset(
                path=self.dataset_cfg["name"],
                split=self.dataset_cfg.get("split", "train"),
                cache_dir=self.dataset_cfg.get("cache_dir"),
            )

            max_train_samples = self.dataset_cfg.get("max_train_samples")
            if max_train_samples is not None:
                max_train_samples = min(int(max_train_samples), len(dataset))
                dataset = dataset.shuffle(seed=self.seed).select(range(max_train_samples))

            self.dataset = dataset.with_transform(self._transform_examples)

    @staticmethod
    def collate_fn(examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

        input_ids = torch.stack([example["input_ids"] for example in examples])

        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
        }

    def train_dataloader(self) -> DataLoader:
        if self.dataset is None:
            raise RuntimeError("DataModule is not set up. Call setup('fit') before requesting dataloader.")

        return DataLoader(
            self.dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=int(self.dataset_cfg.get("num_workers", 4)),
            collate_fn=self.collate_fn,
            pin_memory=True,
            drop_last=True,
        )
