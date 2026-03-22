from typing import Any, Dict, List, Optional, Union

from datasets import Dataset
from PIL import Image
from torch.utils.data import Dataset as TorchDataset
from torchvision import transforms
from transformers import CLIPTokenizer


def _convert_rgb(image: Image.Image) -> Image.Image:
    return image.convert("RGB")


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

