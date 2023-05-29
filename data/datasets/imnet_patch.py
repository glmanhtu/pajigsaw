import logging
import math
import os
import random
from enum import Enum
from typing import Callable, Optional, Union

import torch
import torchvision
from PIL import Image
from datasets import load_dataset
from torchvision.datasets import VisionDataset

logger = logging.getLogger("pajisaw")
_Target = int


class _Split(Enum):
    TRAIN = "train"
    VAL = "validation"
    TEST = "test"  # NOTE: torchvision does not support the test split

    @property
    def length(self) -> float:
        split_lengths = {
            _Split.TRAIN: 0.8,  # percentage of the dataset
            _Split.VAL: 0.1,
            _Split.TEST: 0.1,
        }
        return split_lengths[self]

    @staticmethod
    def from_string(name):
        for key in _Split:
            if key.value == name:
                return key


class ImNetPatch(VisionDataset):
    Target = Union[_Target]
    Split = Union[_Split]

    def __init__(
        self,
        root: str,
        split: "ImNetPatch.Split",
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        image_size=224,
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)
        self._split = split
        self.dataset = load_dataset("imagenet-1k", split=split.value, cache_dir=root)
        self.dataset_path = os.path.join(root, split.value)

        self.entries = {}
        self.entry_id_map = {}
        self.image_size = image_size

        self.cropper_class = torchvision.transforms.RandomCrop
        if split != _Split.TRAIN:
            self.cropper_class = torchvision.transforms.CenterCrop

    @property
    def split(self) -> "ImNetPatch.Split":
        return self._split

    def load_entries(self):
        pass

    def generate_entries(self):
        pass

    def __getitem__(self, index: int):
        if len(self.entries) == 0:
            self.load_entries()

        image = self.dataset[index]['image'].convert('RGB')
        gap = random.randint(1, 15)
        ratio_short = self.image_size / min(image.width, image.height)
        ratio_long = (self.image_size * 2 + gap) / max(image.width, image.height)
        ratio = max(ratio_long, ratio_short)
        if ratio > 1:
            image = image.resize((math.ceil(ratio * image.width), math.ceil(ratio * image.height)), Image.LANCZOS)
        if image.width > image.height:
            cropper = self.cropper_class((self.image_size, self.image_size * 2 + gap))
            patch = cropper(image)
            first_img = patch.crop((0, 0, self.image_size, self.image_size))
            second_img = patch.crop((self.image_size + gap, 0, self.image_size * 2 + gap, self.image_size))
        else:
            cropper = self.cropper_class((self.image_size * 2 + gap, self.image_size))
            patch = cropper(image)
            first_img = patch.crop((0, 0, self.image_size, self.image_size))
            second_img = patch.crop((0, self.image_size + gap, self.image_size, self.image_size * 2 + gap))

        label = 1.
        if 0.5 < torch.rand(1):
            tmp = first_img
            first_img = second_img
            second_img = tmp
            label = 0.

        if self.transform is not None:
            first_img, second_img = self.transform(first_img, second_img)

        assert isinstance(first_img, torch.Tensor)
        assert isinstance(second_img, torch.Tensor)

        stacked_img = torch.stack([first_img, second_img], dim=0)
        return stacked_img, torch.tensor(label, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.dataset)

