from collections import defaultdict
from typing import Any, DefaultDict, Dict, List, Optional, Tuple

import pytorch_lightning as pl
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset as TorchDataset
from transformers import CLIPTokenizer

from src.utils.kohya import (
    AspectRatioBucketBatchSampler,
    HuggingFaceImageTextDataset,
    KohyaStyleDataset,
)


class StableDiffusionDataModule(pl.LightningDataModule):
    """Training DataModule supporting HF datasets and local Kohya-style datasets."""

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

        self.dataset: Optional[TorchDataset] = None
        self.batch_sampler: Optional[AspectRatioBucketBatchSampler] = None

    def prepare_data(self) -> None:
        if self.dataset_cfg.get("name"):
            load_dataset(
                path=self.dataset_cfg["name"],
                split=self.dataset_cfg.get("split", "train"),
                cache_dir=self.dataset_cfg.get("cache_dir"),
            )

    def _build_hf_dataset(self) -> HuggingFaceImageTextDataset:
        resolution = int(self.dataset_cfg.get("resolution", 512))
        center_crop = self.dataset_cfg.get("center_crop", True)
        random_flip = self.dataset_cfg.get("random_flip", True)

        dataset = load_dataset(
            path=self.dataset_cfg["name"],
            split=self.dataset_cfg.get("split", "train"),
            cache_dir=self.dataset_cfg.get("cache_dir"),
        )

        max_train_samples = self.dataset_cfg.get("max_train_samples")
        if max_train_samples is not None:
            max_train_samples = min(int(max_train_samples), len(dataset))
            dataset = dataset.shuffle(seed=self.seed).select(range(max_train_samples))

        return HuggingFaceImageTextDataset(
            dataset=dataset,
            tokenizer=self.tokenizer,
            image_column=self.image_column,
            caption_column=self.caption_column,
            resolution=resolution,
            center_crop=center_crop,
            random_flip=random_flip,
        )

    def _normalize_local_dataset_configs(self) -> List[Dict[str, Any]]:
        raw_datasets = self.dataset_cfg.get("datasets")
        if isinstance(raw_datasets, list) and raw_datasets:
            normalized: List[Dict[str, Any]] = []
            for item in raw_datasets:
                if isinstance(item, dict) and "kwargs" in item and isinstance(item["kwargs"], dict):
                    merged = dict(item)
                    kwargs = dict(merged.pop("kwargs"))
                    kwargs.update(merged)
                    kwargs.pop("class", None)
                    normalized.append(kwargs)
                elif isinstance(item, dict):
                    local_item = dict(item)
                    local_item.pop("class", None)
                    normalized.append(local_item)
            return normalized

        if self.dataset_cfg.get("dataset_folder"):
            return [dict(self.dataset_cfg)]

        return []

    def _build_local_dataset(self) -> TorchDataset:
        dataset_items = self._normalize_local_dataset_configs()
        if not dataset_items:
            raise ValueError("No local dataset configuration found.")

        use_arb = self.dataset_cfg.get("use_arb", True)
        resolution = int(self.dataset_cfg.get("resolution", self.dataset_cfg.get("size", 512)))
        shared_arb_config = self.dataset_cfg.get("arb_config", {})
        shared_cache_dir = self.dataset_cfg.get("cache_dir")

        datasets: List[KohyaStyleDataset] = []
        merged_bucket_indices: DefaultDict[Tuple[int, int], List[int]] = defaultdict(list)
        offset = 0
        for item in dataset_items:
            item_cfg = dict(self.dataset_cfg)
            item_cfg.update(item)
            item_cfg.pop("datasets", None)
            item_cfg.pop("name", None)
            item_cfg.pop("split", None)
            item_cfg["size"] = int(item_cfg.get("size", resolution))
            item_cfg["use_arb"] = item_cfg.get("use_arb", use_arb)
            item_cfg["arb_config"] = item_cfg.get("arb_config", shared_arb_config)
            item_cfg["cache_dir"] = item_cfg.get("cache_dir", shared_cache_dir)
            item_cfg["image_column"] = item_cfg.get("image_column", self.image_column)
            item_cfg["caption_column"] = item_cfg.get("caption_column", self.caption_column)
            item_cfg.pop("num_workers", None)
            item_cfg.pop("max_train_samples", None)
            item_cfg.pop("resolution", None)
            item_cfg.pop("batch_size", None)
            dataset = KohyaStyleDataset(tokenizer=self.tokenizer, **item_cfg)
            datasets.append(dataset)
            for bucket, indices in dataset.bucket_indices.items():
                merged_bucket_indices[bucket].extend([offset + idx for idx in indices])
            offset += len(dataset)

        concatenated = torch.utils.data.ConcatDataset(datasets)
        if use_arb:
            self.batch_sampler = AspectRatioBucketBatchSampler(
                buckets=dict(merged_bucket_indices),
                batch_size=self.train_batch_size,
                drop_last=True,
                shuffle=True,
                seed=self.seed,
            )
        else:
            self.batch_sampler = None
        return concatenated

    def setup(self, stage: Optional[str] = None) -> None:
        if stage not in (None, "fit"):
            return

        if self.dataset_cfg.get("name"):
            self.dataset = self._build_hf_dataset()
            self.batch_sampler = None
            return

        self.dataset = self._build_local_dataset()

    @staticmethod
    def collate_fn(examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        input_ids = torch.stack([example["input_ids"] for example in examples])

        batch: Dict[str, Any] = {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
        }

        if "caption" in examples[0]:
            batch["captions"] = [example["caption"] for example in examples]
        if "bucket" in examples[0]:
            batch["bucket"] = torch.stack([example["bucket"] for example in examples])
        if "original_size" in examples[0]:
            batch["original_size"] = torch.stack([example["original_size"] for example in examples])
        if "crop_top_left" in examples[0]:
            batch["crop_top_left"] = torch.stack([example["crop_top_left"] for example in examples])
        if "target_size" in examples[0]:
            batch["target_size"] = torch.stack([example["target_size"] for example in examples])
        return batch

    def train_dataloader(self) -> DataLoader:
        if self.dataset is None:
            raise RuntimeError("DataModule is not set up. Call setup('fit') before requesting dataloader.")

        num_workers = int(self.dataset_cfg.get("num_workers", 4))
        persistent_workers = self.dataset_cfg.get("persistent_workers", False) and num_workers > 0
        pin_memory = self.dataset_cfg.get("pin_memory", True)

        if self.batch_sampler is not None:
            return DataLoader(
                self.dataset,
                batch_sampler=self.batch_sampler,
                num_workers=num_workers,
                collate_fn=self.collate_fn,
                pin_memory=pin_memory,
                persistent_workers=persistent_workers,
            )

        return DataLoader(
            self.dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=self.collate_fn,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            drop_last=True,
        )
