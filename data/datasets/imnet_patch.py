import logging
import math
import os
import numpy as np
from enum import Enum
from typing import Callable, Optional, Union

import torch
import torchvision
from PIL import Image
from datasets import load_dataset
from torchvision.datasets import VisionDataset

from data import transforms

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
        with_negative=False
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)
        self._split = split
        self.dataset = load_dataset("imagenet-1k", split=split.value, cache_dir=root)
        self.dataset_path = os.path.join(root, split.value)

        self.entries = {}
        self.entry_id_map = {}
        self.image_size = image_size
        self.with_negative = with_negative

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
        gap = 30
        ratio = (self.image_size * 3 + gap) / min(image.width, image.height)
        if ratio > 1:
            image = image.resize((math.ceil(ratio * image.width), math.ceil(ratio * image.height)), Image.LANCZOS)
        cropper = self.cropper_class((self.image_size * 2 + gap * 2, self.image_size * 2 + gap * 2))
        patch = cropper(image)

        # Crop the image into a grid of 2 x 2 patches
        crops = list(transforms.crop(patch, 2, 2))
        cropper = self.cropper_class(self.image_size)
        first_img = cropper(crops[0])

        # Second image is next to the first image
        second_img = cropper(crops[1])

        # Third image is right below the second image
        third_img = cropper(crops[3])

        # Fourth mage is right below the first image
        fourth_img = cropper(crops[2])

        # For now, the second image connect forward to first image, and backward to third image
        # The first and third images have no connection
        label = 0

        if 0.5 < torch.rand(1):
            # Swap second and four patches, the connection is still forwarding from the second to the first
            second_img, fourth_img = fourth_img, second_img
            label = 1

        if 0.5 < torch.rand(1):
            first_img, second_img = second_img, first_img
            # When we swap the first and second image, then we also need to replace the third by the four image
            # to ensure that the first image have no connection to the third image
            third_img, fourth_img = fourth_img, third_img
            if label == 0:
                label = 2
            else:
                label = 3

        if self.with_negative and 0.2 > torch.rand(1):
            # Negative pair for evaluation
            second_img, third_img = third_img, second_img
            label = 4

        if self.transform is not None:
            first_img, second_img = self.transform(first_img, second_img)

        assert isinstance(first_img, torch.Tensor)
        assert isinstance(second_img, torch.Tensor)

        stacked_img = torch.stack([first_img, second_img], dim=0)
        return stacked_img, torch.tensor(label, dtype=torch.int64)

    def __len__(self) -> int:
        return len(self.dataset)

