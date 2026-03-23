from pathlib import Path
from typing import Any, Dict, List, Optional

import pytorch_lightning as pl
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset as TorchDataset
from torchvision import transforms
from transformers import CLIPTokenizer

from src.utils.data_pipeline import (
    CaptionDataset,
    KohyaRepeatDataset,
    LatentCache,
    LatentCacheDataset,
    RepeatDataset,
    build_latent_cache_namespace,
    cache_all_latents,
    create_dataloader,
    detect_kohya_structure,
)
from src.utils.dataset import HuggingFaceImageTextDataset


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
        self.latent_cache: Optional[LatentCache] = None

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

    def _normalize_local_dataset_configs(self, raw_datasets) -> List[Dict[str, Any]]:
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
        dataset_items = self._normalize_local_dataset_configs(self.dataset_cfg.get("datasets", []))
        if not dataset_items:
            raise ValueError("No local dataset configuration found.")

        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

        datasets: List[TorchDataset] = []
        for item_cfg in dataset_items:
            data_dir = item_cfg["dataset_folder"]

            dataset = CaptionDataset(
                data_dir=data_dir,
                resolution=item_cfg["resolution"],
                caption_extension=item_cfg.get("caption_extension", ".txt"),
                enable_bucket=item_cfg.get("use_bucket", True),
                min_bucket_reso=item_cfg["min_bucket_reso"],
                max_bucket_reso=item_cfg["max_bucket_reso"],
                bucket_reso_steps=item_cfg.get("bucket_reso_steps", 64),
                max_bucket_aspect_ratio=item_cfg.get("max_bucket_aspect_ratio", 2.0),
                shuffle_caption=item_cfg.get("shuffle_caption", False),
                keep_tokens=item_cfg.get("keep_tokens", 0),
                keep_tokens_separator=item_cfg.get("keep_tokens_separator", "|||"),
                group_separator=item_cfg.get("group_separator", "%%"),
                group_shuffle=item_cfg.get("group_shuffle", False),
                caption_group_dropout_rate=item_cfg.get("group_dropout_rate", 0.0),
                tag_separator=item_cfg.get("tag_separator", ", "),
                output_separator=item_cfg.get("output_separator", ", "),
                caption_tag_dropout_rate=item_cfg.get("tag_dropout_rate", 0.0),
                flip_augment=item_cfg.get("random_flip", False),
                transform=transform,
            )
            dataset.shuffle_caption_per_epoch = False

            working_dataset: TorchDataset = dataset

            if item_cfg.get("cache_latents", False):
                latent_cache = self._maybe_build_latent_cache(item_cfg, dataset)
                if latent_cache is not None:
                    working_dataset = LatentCacheDataset(dataset, latent_cache)

            repeats = item_cfg.get("repeats", 1)
            if detect_kohya_structure(data_dir):
                working_dataset = KohyaRepeatDataset(working_dataset)
            elif repeats > 1:
                working_dataset = RepeatDataset(working_dataset, repeats)

            datasets.append(working_dataset)

        if len(datasets) == 1:
            return datasets[0]
        return torch.utils.data.ConcatDataset(datasets)

    def _maybe_build_latent_cache(self, item_cfg: Dict[str, Any], dataset: CaptionDataset) -> Optional[LatentCache]:
        trainer = getattr(self, "trainer", None)
        model = getattr(trainer, "lightning_module", None) if trainer is not None else None
        vae = getattr(model, "vae", None) if model is not None else None
        if vae is None:
            return None

        cache_dir = item_cfg.get("cache_dir") or str(Path(item_cfg["dataset_folder"]) / ".latent_cache")
        cache_namespace, cache_meta = build_latent_cache_namespace(
            {
                "resolution": item_cfg["resolution"],
                "min_reso": item_cfg["min_bucket_reso"],
                "max_reso": item_cfg["max_bucket_reso"],
                "reso_step": item_cfg.get("bucket_reso_steps", 64),
                "max_ar": item_cfg.get("max_bucket_aspect_ratio", 2.0),
                "transformer": self.dataset_cfg.get("transformer", ""),
                "vae": self.dataset_cfg.get("vae", ""),
            }
        )

        latent_cache = LatentCache(
            cache_dir=str(cache_dir),
            vae=vae,
            device=model.device,
            dtype=next(vae.parameters()).dtype,
            cache_namespace=cache_namespace,
            cache_meta=cache_meta,
        )

        cache_all_latents(dataset, vae, latent_cache, model.device, next(vae.parameters()).dtype)
        self.latent_cache = latent_cache
        return latent_cache

    def setup(self, stage: Optional[str] = None) -> None:
        if stage not in (None, "fit"):
            return

        if self.dataset_cfg.get("name"):
            self.dataset = self._build_hf_dataset()
            return

        self.dataset = self._build_local_dataset()

    def collate_fn(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        if "pixel_values" in examples[0]:
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
            return batch

        images = torch.stack([example["image"] for example in examples])
        images = images.to(memory_format=torch.contiguous_format).float()
        tokenized = self.tokenizer(
            [example.get("caption", "") for example in examples],
            truncation=True,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )

        batch = {
            "pixel_values": images,
            "input_ids": tokenized.input_ids,
            "captions": [example.get("caption", "") for example in examples],
            "original_sizes": [example.get("original_size") for example in examples],
            "target_sizes": [example.get("target_size") for example in examples],
            "crop_coords": [example.get("crop_coords") for example in examples],
        }
        if "latent" in examples[0]:
            batch["latent"] = [example["latent"] for example in examples]
            batch["use_cached_latent"] = [example.get("use_cached_latent", False) for example in examples]
        return batch

    def train_dataloader(self) -> DataLoader:
        if self.dataset is None:
            raise RuntimeError("DataModule is not set up. Call setup('fit') before requesting dataloader.")

        num_workers = int(self.dataset_cfg.get("num_workers", 4))
        persistent_workers = self.dataset_cfg.get("persistent_workers", False) and num_workers > 0
        pin_memory = self.dataset_cfg.get("pin_memory", True)
        if self.dataset_cfg.get("name"):
            return create_dataloader(
                self.dataset,
                batch_size=self.train_batch_size,
                shuffle=True,
                num_workers=num_workers,
                collate_fn=self.collate_fn,
                pin_memory=pin_memory,
                persistent_workers=persistent_workers,
                drop_last=True,
                seed=self.seed,
            )

        return create_dataloader(
            self.dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=self.collate_fn,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            drop_last=True,
            seed=self.seed,
        )
