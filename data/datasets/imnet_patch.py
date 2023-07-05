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
        erosion_ratio=0.07,
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
        self.erosion_ratio = erosion_ratio

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
        gap = int(self.image_size * self.erosion_ratio)

        # Resize the image if it does not fit the patch size that we want
        ratio = (self.image_size * 4 + gap * 3) / min(image.width, image.height)
        if ratio > 1:
            image = image.resize((math.ceil(ratio * image.width), math.ceil(ratio * image.height)), Image.LANCZOS)
        cropper = self.cropper_class((self.image_size * 4 + gap * 3, self.image_size * 4 + gap * 3))
        patch = cropper(image)

        # Crop the image into a grid of 4 x 4 patches
        crops = list(transforms.crop(patch, 4, 4))
        cropper = self.cropper_class(self.image_size)
        first_img = cropper(crops[1])

        # Second image is next to the first image
        second_img = cropper(crops[2])

        # Third image is right below the second image
        third_img = cropper(crops[6])

        # Fourth mage is right below the first image
        fourth_img = cropper(crops[5])

        # For now, the second image connect forward to first image, and backward to third image
        # The first and third images have no connection
        label = [1., 0., 0., 0.]
        if self.with_negative and 0.4 > torch.rand(1):
            if 0.5 < torch.rand(1):
                # Negative pair for evaluation
                second_img, third_img = third_img, second_img
            else:
                second_img = cropper(crops[3])

            if 0.5 < torch.rand(1):
                second_img, first_img = first_img, second_img

            label = [0., 0., 0., 0.]

        else:
            if 0.5 < torch.rand(1):
                second_img, fourth_img = fourth_img, second_img
                label = [0., 1., 0., 0.]

            if 0.5 < torch.rand(1):
                first_img, second_img = second_img, first_img
                if label[0] == 1:
                    label = [0., 0., 1., 0.]
                else:
                    label = [0., 0., 0., 1.]

        if self.transform is not None:
            first_img, second_img = self.transform(first_img, second_img)

        assert isinstance(first_img, torch.Tensor)
        assert isinstance(second_img, torch.Tensor)

        stacked_img = torch.stack([first_img, second_img], dim=0)
        return stacked_img, torch.tensor(label, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.dataset)

