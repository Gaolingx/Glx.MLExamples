from __future__ import annotations

import hashlib
import json
import logging
import random
import re
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning.utilities.rank_zero import rank_zero_info, rank_zero_warn


def _stable_caption_hash_int(text: str) -> int:
    if not text:
        return 0
    d = hashlib.md5(text.encode("utf-8")).digest()
    return int.from_bytes(d[:8], byteorder="little", signed=False)


class BucketManager:
    def __init__(
            self,
            base_resolution=1024,
            min_resolution=512,
            max_resolution=2048,
            resolution_step=64,
            max_aspect_ratio=2.0,
    ):
        self.base_resolution = base_resolution
        self.min_resolution = min_resolution
        self.max_resolution = max_resolution
        self.resolution_step = resolution_step
        self.max_aspect_ratio = None if (max_aspect_ratio is None or max_aspect_ratio <= 0) else float(max_aspect_ratio)
        self.buckets = self._generate_buckets()

    def _generate_buckets(self):
        buckets = []
        base_area = self.base_resolution ** 2
        for w in range(self.min_resolution, self.max_resolution + 1, self.resolution_step):
            for h in range(self.min_resolution, self.max_resolution + 1, self.resolution_step):
                area = w * h
                if abs(area - base_area) / base_area > 0.1:
                    continue
                aspect = max(w / h, h / w)
                if self.max_aspect_ratio is not None and aspect > self.max_aspect_ratio:
                    continue
                buckets.append((w, h))
        return buckets

    def get_bucket(self, width, height):
        aspect = width / height
        best_bucket = (self.base_resolution, self.base_resolution)
        best_diff = float("inf")
        for bw, bh in self.buckets:
            diff = abs(aspect - (bw / bh))
            if diff < best_diff:
                best_diff = diff
                best_bucket = (bw, bh)
        return best_bucket


class LatentCache:
    def __init__(self, cache_dir: str, vae, device, dtype, *, cache_namespace: str = "legacy", cache_meta: dict | None = None):
        self.cache_root = Path(cache_dir)
        self.cache_root.mkdir(parents=True, exist_ok=True)
        ns = re.sub(r"[^0-9A-Za-z._-]", "_", str(cache_namespace or "legacy"))
        self.cache_namespace = ns or "legacy"
        self.cache_dir = self.cache_root / self.cache_namespace
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.vae = vae
        self.device = device
        self.dtype = dtype
        self.logger = logging.getLogger(__name__)
        if cache_meta is not None:
            try:
                meta_path = self.cache_dir / "_cache_meta.json"
                meta_path.write_text(
                    json.dumps(cache_meta, ensure_ascii=False, sort_keys=True, indent=2),
                    encoding="utf-8",
                )
            except Exception as exc:
                self.logger.warning("Failed to write cache metadata: %s", exc)

    def _get_cache_path(self, image_path: str) -> Path:
        hash_key = hashlib.md5(image_path.encode()).hexdigest()
        return self.cache_dir / f"{hash_key}.pt"

    def has_cache(self, image_path: str) -> bool:
        return self._get_cache_path(image_path).exists()

    def load_cache(self, image_path: str) -> torch.Tensor:
        cache_path = self._get_cache_path(image_path)
        return torch.load(cache_path, map_location="cpu")

    def save_cache(self, image_path: str, latent: torch.Tensor):
        cache_path = self._get_cache_path(image_path)
        torch.save(latent.cpu(), cache_path)


class CaptionDataset(Dataset):
    SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

    def __init__(
            self,
            data_dir,
            resolution=1024,
            caption_extension=".txt",
            enable_bucket=True,
            min_bucket_reso=512,
            max_bucket_reso=2048,
            bucket_reso_steps=64,
            max_bucket_aspect_ratio=2.0,
            shuffle_caption=False,
            shuffle_caption_per_epoch=False,
            keep_tokens=0,
            keep_tokens_separator="",
            group_separator="",
            group_shuffle=False,
            caption_group_dropout_rate=0.0,
            tag_separator=None,
            output_separator=", ",
            caption_tag_dropout_rate=0.0,
            flip_augment=False,
            transform=None,
    ):
        self.data_dir = Path(data_dir)
        self.resolution = resolution
        self.caption_extension = caption_extension
        self.shuffle_caption = shuffle_caption
        self.shuffle_caption_per_epoch = shuffle_caption_per_epoch
        self._epoch = 0
        self._caption_seed = 0
        self.keep_tokens = keep_tokens
        self.keep_tokens_separator = keep_tokens_separator
        self.group_separator = group_separator or ""
        self.group_shuffle = bool(group_shuffle)
        self.caption_group_dropout_rate = max(0.0, min(1.0, float(caption_group_dropout_rate or 0.0)))
        self.tag_separator = tag_separator
        self.output_separator = output_separator or ", "
        self.caption_tag_dropout_rate = max(0.0, min(1.0, float(caption_tag_dropout_rate or 0.0)))
        self.flip_augment = flip_augment
        self.transform = transform

        self.logger = logging.getLogger(__name__)

        if enable_bucket:
            self.bucket_manager = BucketManager(
                base_resolution=resolution,
                min_resolution=min_bucket_reso,
                max_resolution=max_bucket_reso,
                resolution_step=bucket_reso_steps,
                max_aspect_ratio=max_bucket_aspect_ratio,
            )
        else:
            self.bucket_manager = None

        self.samples = self._scan_dataset()
        self.logger.info("Found %d samples in %s", len(self.samples), data_dir)

    def set_epoch(self, epoch: int, *, seed: int = 0):
        self._epoch = int(epoch)
        self._caption_seed = int(seed)

    def _parse_kohya_dir_name(self, dir_name: str) -> tuple:
        match = re.match(r"^(\d+)_(.+)$", dir_name)
        if match:
            return int(match.group(1)), match.group(2)
        return 1, None

    def _scan_dataset(self):
        from PIL import Image

        samples = []
        if not self.data_dir.exists():
            self.logger.warning("Dataset folder not found: %s", self.data_dir)
            return samples

        paths = sorted(self.data_dir.rglob("*"), key=lambda p: p.as_posix())
        for img_path in paths:
            try:
                if not img_path.is_file() or img_path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
                    continue
                parent_dir = img_path.parent
                if parent_dir != self.data_dir:
                    repeat, concept = self._parse_kohya_dir_name(parent_dir.name)
                else:
                    repeat, concept = 1, None

                sample = self._process_image_file(img_path, repeat, concept)
                if sample:
                    samples.append(sample)
            except Exception as exc:
                self.logger.warning("Error processing sample %s: %s", img_path, exc)
                continue
        return samples

    def _process_image_file(self, img_path: Path, repeat: int = 1, concept: str | None = None) -> dict | None:
        from PIL import Image

        caption_path = img_path.with_suffix(self.caption_extension)
        if not caption_path.exists():
            caption_path = img_path.with_suffix(".txt")

        caption = ""
        if caption_path.exists():
            try:
                caption = caption_path.read_text(encoding="utf-8").strip()
            except Exception as exc:
                self.logger.warning("Failed to read caption for %s: %s", img_path, exc)

        try:
            with Image.open(img_path) as img:
                original_size = img.size
        except Exception as exc:
            self.logger.warning("Failed to open %s: %s", img_path, exc)
            return None

        if self.bucket_manager:
            target_size = self.bucket_manager.get_bucket(*original_size)
        else:
            target_size = (self.resolution, self.resolution)

        crop_coords = self._calculate_crop(original_size, target_size)
        return {
            "image_path": str(img_path),
            "caption": caption,
            "original_size": original_size,
            "target_size": target_size,
            "crop_coords": crop_coords,
            "repeat": repeat,
            "concept": concept,
        }

    @staticmethod
    def _calculate_crop(original_size, target_size):
        ow, oh = original_size
        tw, th = target_size
        scale = max(tw / ow, th / oh)
        scaled_w = int(ow * scale)
        scaled_h = int(oh * scale)
        left = (scaled_w - tw) // 2
        top = (scaled_h - th) // 2
        return (left, top, left + tw, top + th)

    def _process_caption(self, caption):
        if not caption:
            return ""

        rng = random
        if self.shuffle_caption_per_epoch:
            seed_value = self._caption_seed ^ self._epoch ^ _stable_caption_hash_int(caption)
            rng = random.Random(seed_value)

        lines = [line.strip() for line in caption.splitlines() if line.strip()]
        if lines:
            tags_text = lines[0]
            nl_text = " ".join(lines[1:]).strip()
        else:
            tags_text = caption.strip()
            nl_text = ""

        tag_separator = self.tag_separator
        if tag_separator is None:
            tag_separator = "," if "," in tags_text else None

        def _split_tags(text: str):
            if not text:
                return []
            if tag_separator:
                return [t.strip() for t in text.split(tag_separator) if t.strip()]
            return [t.strip() for t in text.split() if t.strip()]

        def _drop_tokens(tokens):
            if self.caption_tag_dropout_rate <= 0.0 or not tokens:
                return tokens
            kept = [t for t in tokens if rng.random() >= self.caption_tag_dropout_rate]
            if not kept:
                kept = [rng.choice(tokens)]
            return kept

        fixed_tokens = []
        flex_tokens = []
        fixed_suffix_tokens = []

        if self.keep_tokens_separator and self.keep_tokens_separator in tags_text:
            fixed_part, flex_part = tags_text.split(self.keep_tokens_separator, 1)
            if self.keep_tokens_separator in flex_part:
                flex_part, fixed_suffix_part = flex_part.split(self.keep_tokens_separator, 1)
                fixed_suffix_tokens = _split_tags(fixed_suffix_part)

            fixed_tokens = _split_tags(fixed_part)
            flex_tokens = _split_tags(flex_part)
        else:
            tokens = _split_tags(tags_text)
            if self.keep_tokens > 0:
                fixed_tokens = tokens[: self.keep_tokens]
                flex_tokens = tokens[self.keep_tokens:]
            else:
                flex_tokens = tokens

        if self.group_separator:
            grouped_tokens = []
            for group_text in self.group_separator.join(flex_tokens).split(self.group_separator):
                tokens = _split_tags(group_text)
                if tokens:
                    grouped_tokens.append(tokens)

            if self.group_shuffle and len(grouped_tokens) > 1:
                rng.shuffle(grouped_tokens)

            processed_groups = []
            for tokens in grouped_tokens:
                if self.caption_group_dropout_rate > 0.0 and rng.random() < self.caption_group_dropout_rate:
                    continue
                tokens = _drop_tokens(tokens)
                if self.shuffle_caption and len(tokens) > 1:
                    rng.shuffle(tokens)
                if tokens:
                    processed_groups.append(tokens)

            flex_tokens = [token for group in processed_groups for token in group]
        else:
            if self.shuffle_caption and len(flex_tokens) > 1:
                rng.shuffle(flex_tokens)
            flex_tokens = _drop_tokens(flex_tokens)

        tags = fixed_tokens + flex_tokens + fixed_suffix_tokens
        tags_out = self.output_separator.join(tags)
        if tags_out and nl_text:
            return f"{tags_out}{self.output_separator}{nl_text}"
        if nl_text:
            return nl_text
        return tags_out

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        import numpy as np
        from PIL import Image

        sample = self.samples[idx]
        image_path = sample["image_path"]
        try:
            with Image.open(image_path) as img:
                image = img.convert("RGB")
        except Exception as exc:
            raise RuntimeError(f"Failed to load image idx={idx} path={image_path}: {exc}") from exc

        tw, th = sample["target_size"]
        scale = max(tw / image.width, th / image.height)
        new_w = int(image.width * scale)
        new_h = int(image.height * scale)
        image = image.resize((new_w, new_h), Image.LANCZOS)

        left = (new_w - tw) // 2
        top = (new_h - th) // 2
        image = image.crop((left, top, left + tw, top + th))

        if self.flip_augment and random.random() > 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)

        caption = self._process_caption(sample["caption"])

        if self.transform:
            image = self.transform(image)
        else:
            image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
            image = image * 2.0 - 1.0

        return {
            "image": image,
            "caption": caption,
            "original_size": sample["original_size"],
            "target_size": sample["target_size"],
            "crop_coords": sample["crop_coords"],
            "repeat": sample.get("repeat", 1),
        }


def _seed_worker(worker_id: int):
    worker_seed = int(torch.initial_seed() % (2 ** 32))
    random.seed(worker_seed)
    try:
        import numpy as np

        np.random.seed(worker_seed)
    except Exception:
        pass
    try:
        torch.manual_seed(worker_seed)
    except Exception:
        pass


def _default_collate_fn(batch):
    images = torch.stack([item["image"] for item in batch])
    captions = [item["caption"] for item in batch]
    original_sizes = [item["original_size"] for item in batch]
    target_sizes = [item["target_size"] for item in batch]
    crop_coords = [item["crop_coords"] for item in batch]
    result = {
        "images": images,
        "captions": captions,
        "original_sizes": original_sizes,
        "target_sizes": target_sizes,
        "crop_coords": crop_coords,
    }
    if "latent" in batch[0]:
        result["latent"] = [item["latent"] for item in batch]
        result["use_cached_latent"] = [item.get("use_cached_latent", False) for item in batch]
    return result


class _ListBatchSampler:
    def __init__(self, batches):
        self.batches = batches

    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)


def create_dataloader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        *,
        seed=None,
        persistent_workers=False,
        collate_fn=None,
        drop_last=False,
):
    def _get_sample_for_index(ds, idx):
        if hasattr(ds, "index_map"):
            base_idx = ds.index_map[idx]
            return ds.dataset.samples[base_idx]
        if hasattr(ds, "repeats") and hasattr(ds, "dataset"):
            base_len = len(ds.dataset)
            return ds.dataset.samples[idx % base_len]
        if hasattr(ds, "samples"):
            return ds.samples[idx]
        return None

    def _build_bucketed_batches(ds, bs, do_shuffle):
        rng = random.Random(int(seed)) if seed is not None else random
        buckets = {}
        for idx in range(len(ds)):
            sample = _get_sample_for_index(ds, idx)
            if not sample:
                continue
            target_size = sample.get("target_size")
            buckets.setdefault(target_size, []).append(idx)
        batches = []
        for indices in buckets.values():
            if do_shuffle:
                rng.shuffle(indices)
            for i in range(0, len(indices), bs):
                batch = indices[i: i + bs]
                if batch:
                    batches.append(batch)
        if do_shuffle:
            rng.shuffle(batches)
        return batches

    if batch_size > 1 and hasattr(dataset, "samples"):
        bucketed_batches = _build_bucketed_batches(dataset, batch_size, shuffle)
        if bucketed_batches:
            return DataLoader(
                dataset,
                batch_sampler=_ListBatchSampler(bucketed_batches),
                num_workers=num_workers,
                pin_memory=pin_memory,
                collate_fn=collate_fn or _default_collate_fn,
                drop_last=drop_last,
                persistent_workers=bool(int(num_workers or 0) > 0 and persistent_workers),
                worker_init_fn=_seed_worker if int(num_workers or 0) > 0 else None,
            )

    g = None
    if seed is not None:
        g = torch.Generator()
        g.manual_seed(int(seed))

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn or _default_collate_fn,
        generator=g,
        drop_last=drop_last,
        persistent_workers=bool(int(num_workers or 0) > 0 and persistent_workers),
        worker_init_fn=_seed_worker if int(num_workers or 0) > 0 else None,
    )


class RepeatDataset(Dataset):
    def __init__(self, dataset, repeats=1):
        self.dataset = dataset
        self.repeats = max(1, int(repeats))

    @property
    def samples(self):
        return self.dataset.samples

    def __len__(self):
        return len(self.dataset) * self.repeats

    def __getitem__(self, idx):
        return self.dataset[idx % len(self.dataset)]


class KohyaRepeatDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.index_map = self._build_index_map()

    @property
    def samples(self):
        return self.dataset.samples

    def _build_index_map(self):
        index_map = []
        for i in range(len(self.dataset)):
            sample = self.dataset.samples[i]
            repeat = sample.get("repeat", 1)
            for _ in range(repeat):
                index_map.append(i)
        return index_map

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        real_idx = self.index_map[idx]
        return self.dataset[real_idx]


class LatentCacheDataset(Dataset):
    def __init__(self, dataset, cache: LatentCache):
        self.dataset = dataset
        self.cache = cache

    @property
    def samples(self):
        return self.dataset.samples

    def __len__(self):
        return len(self.dataset)

    def _resolve_sample_for_index(self, idx: int):
        dataset = self.dataset

        if hasattr(dataset, "index_map") and hasattr(dataset, "dataset"):
            base_idx = dataset.index_map[idx]
            base_dataset = dataset.dataset
            if hasattr(base_dataset, "samples"):
                return base_dataset.samples[base_idx]

        if hasattr(dataset, "repeats") and hasattr(dataset, "dataset"):
            base_dataset = dataset.dataset
            base_len = len(base_dataset)
            if base_len <= 0:
                raise IndexError("Cannot resolve sample from empty repeated dataset.")
            base_idx = idx % base_len
            if hasattr(base_dataset, "samples"):
                return base_dataset.samples[base_idx]

        if hasattr(dataset, "samples"):
            return dataset.samples[idx]

        raise AttributeError("Wrapped dataset does not expose sample metadata required for latent cache lookup.")

    def __getitem__(self, idx):
        item = self.dataset[idx]
        sample = self._resolve_sample_for_index(idx)
        image_path = sample["image_path"]

        if self.cache.has_cache(image_path):
            item["latent"] = self.cache.load_cache(image_path)
            item["use_cached_latent"] = True
        else:
            item["use_cached_latent"] = False
        return item


def detect_kohya_structure(data_dir: str) -> bool:
    data_path = Path(data_dir)
    if not data_path.exists():
        return False

    for subdir in data_path.iterdir():
        if subdir.is_dir():
            match = re.match(r"^(\d+)_(.+)$", subdir.name)
            if match:
                return True
    return False


def build_latent_cache_namespace(args: dict | object) -> tuple[str, dict]:
    def _get(name: str, default):
        if isinstance(args, dict):
            return args.get(name, default)
        return getattr(args, name, default)

    payload = {
        "cache_schema": "anima_latent_v2",
        "resolution": int(_get("resolution", 1024)),
        "min_reso": int(_get("min_reso", 512)),
        "max_reso": int(_get("max_reso", 2048)),
        "reso_step": int(_get("reso_step", 64)),
        "max_ar": float(_get("max_ar", 2.0)),
        "transformer": str(_get("transformer", "")),
        "vae": str(_get("vae", "")),
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    digest = hashlib.md5(raw.encode("utf-8")).hexdigest()[:16]
    namespace = f"cfg_{digest}"
    return namespace, payload


def cache_all_latents(dataset, vae, cache: LatentCache, device, dtype, progress_log_interval=10):
    import numpy as np
    from PIL import Image

    rank_zero_info("Starting latent cache generation...")
    cached_count = 0
    skipped_count = 0
    failed_count = 0
    total = len(dataset.samples)

    def _log_progress(processed_count: int):
        rank_zero_info(
            "Latent caching progress: %d/%d (new=%d skip=%d fail=%d)",
            processed_count,
            total,
            cached_count,
            skipped_count,
            failed_count,
        )

    try:
        for i, sample in enumerate(dataset.samples):
            image_path = sample["image_path"]

            if cache.has_cache(image_path):
                skipped_count += 1
                if ((i + 1) % progress_log_interval == 0) or ((i + 1) == total):
                    _log_progress(i + 1)
                continue

            try:
                with Image.open(image_path) as img:
                    image = img.convert("RGB")
                tw, th = sample["target_size"]
                scale = max(tw / image.width, th / image.height)
                new_w = int(image.width * scale)
                new_h = int(image.height * scale)
                image = image.resize((new_w, new_h), Image.LANCZOS)
                left = (new_w - tw) // 2
                top = (new_h - th) // 2
                image = image.crop((left, top, left + tw, top + th))

                img_tensor = torch.from_numpy(np.array(image))
                img_tensor = img_tensor.permute(2, 0, 1).float() / 255.0
                img_tensor = img_tensor * 2.0 - 1.0
                img_tensor = img_tensor.unsqueeze(0)
                img_tensor = img_tensor.to(device, dtype=dtype)

                with torch.no_grad():
                    latent = vae.encode(img_tensor).latent_dist.sample()
                    latent = latent * vae.config.scaling_factor

                cache.save_cache(image_path, latent.squeeze(0))
                cached_count += 1

            except Exception as exc:
                failed_count += 1
                rank_zero_warn(f"Latent cache failed at idx={i} path={image_path}: {exc}")

            if ((i + 1) % progress_log_interval == 0) or ((i + 1) == total):
                _log_progress(i + 1)
    finally:
        if total > 0 and (total % progress_log_interval) != 0:
            _log_progress(total)

    complete = (cached_count + skipped_count) == total
    rank_zero_info(
        "Latent caching complete: new=%d, skipped=%d, failed=%d (total=%d, complete=%s)",
        cached_count,
        skipped_count,
        failed_count,
        total,
        "yes" if complete else "no",
    )
    return {
        "cached": cached_count,
        "skipped": skipped_count,
        "failed": failed_count,
        "total": total,
        "complete": complete,
    }
