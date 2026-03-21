import math
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, Iterable, List, Optional, Sequence, Tuple

import imagesize
import numpy as np
import torch
from datasets import Dataset, load_dataset
from PIL import Image
from torch.utils.data import BatchSampler, Dataset as TorchDataset
from torchvision import transforms
from transformers import CLIPTokenizer

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _convert_rgb(image: Image.Image) -> Image.Image:
    return image.convert("RGB")


class AspectRatioBucketBatchSampler(BatchSampler):
    """Batch sampler that keeps a batch inside the same aspect-ratio bucket."""

    def __init__(
            self,
            buckets: Dict[Tuple[int, int], List[int]],
            batch_size: int,
            drop_last: bool,
            shuffle: bool,
            seed: int = 42,
    ) -> None:
        self.buckets = {key: list(indices) for key, indices in buckets.items() if indices}
        self.bucket_keys = list(self.buckets.keys())
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self) -> Iterable[List[int]]:
        rng = random.Random(self.seed + self.epoch)
        per_bucket_batches: List[List[int]] = []
        for bucket_key in self.bucket_keys:
            indices = list(self.buckets[bucket_key])
            if self.shuffle:
                rng.shuffle(indices)
            for start in range(0, len(indices), self.batch_size):
                batch = indices[start: start + self.batch_size]
                if len(batch) < self.batch_size and self.drop_last:
                    continue
                per_bucket_batches.append(batch)

        if self.shuffle:
            rng.shuffle(per_bucket_batches)

        for batch in per_bucket_batches:
            yield batch

    def __len__(self) -> int:
        total = 0
        for indices in self.buckets.values():
            if self.drop_last:
                total += len(indices) // self.batch_size
            else:
                total += math.ceil(len(indices) / self.batch_size)
        return total


class HuggingFaceImageTextDataset(TorchDataset):
    def __init__(
            self,
            dataset: Dataset,
            tokenizer: CLIPTokenizer,
            image_column: str,
            caption_column: str,
            resolution: int,
            center_crop: bool,
            random_flip: bool,
    ) -> None:
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.image_column = image_column
        self.caption_column = caption_column

        crop_op = transforms.CenterCrop(resolution) if center_crop else transforms.RandomCrop(resolution)
        flip_op = transforms.RandomHorizontalFlip() if random_flip else transforms.Lambda(lambda x: x)
        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
                crop_op,
                flip_op,
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample = self.dataset[index]
        image = sample[self.image_column]
        caption = sample[self.caption_column]
        if not isinstance(image, Image.Image):
            image = Image.open(image)
        image = _convert_rgb(image)
        pixel_values = self.image_transforms(image)
        tokenized = self.tokenizer(
            caption,
            truncation=True,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )
        return {
            "pixel_values": pixel_values,
            "input_ids": tokenized.input_ids.squeeze(0),
            "caption": caption,
        }


class KohyaStyleDataset(TorchDataset):
    """Local image-text dataset with kohya-style caption processing and ARB support."""

    def __init__(
            self,
            tokenizer: CLIPTokenizer,
            dataset_folder: str,
            size: int = 512,
            image_column: str = "image",
            caption_column: str = "text",
            keep_token_seperator: str = "|||",
            tag_seperator: str = ", ",
            seperator: str = ", ",
            group_seperator: str = "%%",
            tag_shuffle: bool = True,
            group_shuffle: bool = False,
            tag_dropout_rate: float = 0.0,
            group_dropout_rate: float = 0.0,
            use_cached_meta: bool = True,
            meta_postfix: str = "_cached",
            center_crop: bool = False,
            random_flip: bool = False,
            use_arb: bool = True,
            arb_config: Optional[Dict[str, Any]] = None,
            cache_dir: Optional[str] = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.dataset_folder = Path(dataset_folder)
        self.size = int(size)
        self.image_column = image_column
        self.caption_column = caption_column
        self.keep_token_seperator = keep_token_seperator
        self.tag_seperator = tag_seperator
        self.seperator = seperator
        self.group_seperator = group_seperator
        self.tag_shuffle = tag_shuffle
        self.group_shuffle = group_shuffle
        self.tag_dropout_rate = float(tag_dropout_rate)
        self.group_dropout_rate = float(group_dropout_rate)
        self.use_cached_meta = use_cached_meta
        self.meta_postfix = meta_postfix
        self.center_crop = center_crop
        self.random_flip = random_flip
        self.use_arb = use_arb
        self.arb_config = arb_config or {}
        self.cache_dir = Path(cache_dir) if cache_dir else self.dataset_folder
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.to_tensor = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

        self.samples = self._load_samples_metadata()
        self.bucket_resolutions = self._build_bucket_resolutions()
        self.bucket_indices = self._assign_bucket_indices()

    def _metadata_cache_path(self) -> Path:
        sanitized = self.dataset_folder.name or "dataset"
        return self.cache_dir / f"{sanitized}_metadata{self.meta_postfix}.npy"

    def _load_samples_metadata(self) -> List[Dict[str, Any]]:
        cache_path = self._metadata_cache_path()
        if cache_path.exists() and self.use_cached_meta:
            raw = np.load(cache_path, allow_pickle=True)
            return [dict(item) for item in raw.tolist()]

        samples: List[Dict[str, Any]] = []
        for root, _, files in os.walk(self.dataset_folder):
            for file_name in files:
                suffix = Path(file_name).suffix.lower()
                if suffix not in IMAGE_EXTENSIONS:
                    continue
                image_path = Path(root) / file_name
                text_path = image_path.with_suffix(".txt")
                width, height = imagesize.get(str(image_path))
                if width <= 0 or height <= 0:
                    continue
                samples.append(
                    {
                        "image_path": str(image_path),
                        "text_path": str(text_path),
                        "width": int(width),
                        "height": int(height),
                    }
                )

        np.save(cache_path, np.array(samples, dtype=object), allow_pickle=True)
        return samples

    def _build_bucket_resolutions(self) -> List[Tuple[int, int]]:
        if not self.use_arb:
            return [(self.size, self.size)]

        target_res = int(self.arb_config.get("target_res", self.size))
        res_step = int(self.arb_config.get("res_step", 64))
        min_size = int(self.arb_config.get("min_size", res_step))
        max_size = int(self.arb_config.get("max_size", target_res * 2))
        max_area = target_res * target_res

        resolutions = {(target_res, target_res)}
        for width in range(min_size, max_size + 1, res_step):
            for height in range(min_size, max_size + 1, res_step):
                area = width * height
                if area > max_area:
                    continue
                if width < res_step or height < res_step:
                    continue
                resolutions.add((width, height))

        return sorted(resolutions, key=lambda item: (item[0] * item[1], item[0]))

    def _closest_bucket(self, width: int, height: int) -> Tuple[int, int]:
        if not self.use_arb:
            return self.bucket_resolutions[0]

        aspect = width / height
        best_bucket = self.bucket_resolutions[0]
        best_score: Optional[Tuple[float, float]] = None
        for bucket in self.bucket_resolutions:
            bucket_w, bucket_h = bucket
            bucket_aspect = bucket_w / bucket_h
            aspect_error = abs(math.log(aspect) - math.log(bucket_aspect))
            area_error = abs((bucket_w * bucket_h) - min(width * height, self.size * self.size))
            score = (aspect_error, area_error)
            if best_score is None or score < best_score:
                best_score = score
                best_bucket = bucket
        return best_bucket

    def _assign_bucket_indices(self) -> Dict[Tuple[int, int], List[int]]:
        bucket_indices: DefaultDict[Tuple[int, int], List[int]] = defaultdict(list)
        for index, sample in enumerate(self.samples):
            bucket = self._closest_bucket(int(sample["width"]), int(sample["height"]))
            sample["bucket"] = bucket
            bucket_indices[bucket].append(index)
        return dict(bucket_indices)

    def __len__(self) -> int:
        return len(self.samples)

    def _read_caption(self, text_path: str) -> str:
        if not os.path.exists(text_path):
            return ""
        with open(text_path, "r", encoding="utf-8") as handle:
            return handle.read().strip()

    def _split_tags(self, text: str, separator: str) -> List[str]:
        return [part.strip() for part in text.split(separator) if part and part.strip()]

    def _process_caption(self, raw_caption: str) -> str:
        if not raw_caption:
            return ""

        keep_tokens: List[str] = []
        remainder = raw_caption
        if self.keep_token_seperator and self.keep_token_seperator in raw_caption:
            keep_part, remainder = raw_caption.split(self.keep_token_seperator, 1)
            keep_tokens = self._split_tags(keep_part, self.tag_seperator)

        groups = self._split_tags(remainder, self.group_seperator) if self.group_seperator else [remainder]
        if self.group_shuffle:
            random.shuffle(groups)

        merged_tags = list(keep_tokens)
        for group in groups:
            if self.group_dropout_rate > 0.0 and random.random() < self.group_dropout_rate:
                continue

            tags = self._split_tags(group, self.tag_seperator)
            if self.tag_dropout_rate > 0.0:
                tags = [tag for tag in tags if random.random() >= self.tag_dropout_rate]
            if self.tag_shuffle:
                random.shuffle(tags)
            merged_tags.extend(tags)

        return self.seperator.join([tag for tag in merged_tags if tag])

    def _get_crop_coordinates(self, resized_w: int, resized_h: int, target_w: int, target_h: int) -> Tuple[int, int]:
        max_left = max(0, resized_w - target_w)
        max_top = max(0, resized_h - target_h)
        if self.center_crop:
            return max_left // 2, max_top // 2
        crop_left = random.randint(0, max_left) if max_left > 0 else 0
        crop_top = random.randint(0, max_top) if max_top > 0 else 0
        return crop_left, crop_top

    def _load_and_transform_image(self, image_path: str, bucket: Tuple[int, int]) -> torch.Tensor:
        target_w, target_h = bucket
        with Image.open(image_path) as image:
            image = _convert_rgb(image)
            scale = max(target_w / image.width, target_h / image.height)
            resized_w = max(target_w, int(round(image.width * scale)))
            resized_h = max(target_h, int(round(image.height * scale)))
            image = image.resize((resized_w, resized_h), resample=Image.Resampling.BICUBIC)
            crop_left, crop_top = self._get_crop_coordinates(resized_w, resized_h, target_w, target_h)
            image = image.crop((crop_left, crop_top, crop_left + target_w, crop_top + target_h))
            if self.random_flip and random.random() < 0.5:
                image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            tensor = self.to_tensor(image)

        return tensor

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample = self.samples[index]
        image_path = sample["image_path"]
        text_path = sample["text_path"]
        bucket = tuple(sample.get("bucket", (self.size, self.size)))
        caption = self._process_caption(self._read_caption(text_path))
        pixel_values = self._load_and_transform_image(image_path, bucket)
        tokenized = self.tokenizer(
            caption,
            truncation=True,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )
        return {
            "pixel_values": pixel_values,
            "input_ids": tokenized.input_ids.squeeze(0),
            "caption": caption,
            "bucket": torch.tensor([bucket[1], bucket[0]], dtype=torch.long),
        }
