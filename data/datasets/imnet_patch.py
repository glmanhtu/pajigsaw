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
        min_rand, max_rand = 15, 45
        ratio = (self.image_size * 2.5 + max_rand) / min(image.width, image.height)
        if ratio > 1:
            image = image.resize((math.ceil(ratio * image.width), math.ceil(ratio * image.height)), Image.LANCZOS)
        cropper = self.cropper_class((self.image_size * 2 + max_rand, self.image_size * 2 + max_rand))
        patch = cropper(image)
        first_img = patch.crop((0, 0, self.image_size, self.image_size))
        gap_x = random.randint(min_rand, max_rand)
        gap_y = random.randint(min_rand, max_rand)
        second_img = patch.crop((self.image_size + gap_x, gap_y, self.image_size * 2 + gap_x, self.image_size))

        gap = random.randint(min_rand, max_rand)
        # Third image is right below the second image
        third_img = patch.crop((self.image_size + gap, self.image_size + gap, self.image_size * 2 + gap,
                                self.image_size * 2 + gap))

        # Fourth mage is right below the first image
        fourth_img = patch.crop((0, self.image_size + gap, self.image_size, self.image_size * 2 + gap))

        # For now, the second image connect forward to first image, and backward to third image
        # The first and third images have no connection
        label = 1.
        if 0.5 < torch.rand(1):
            tmp = first_img
            first_img = second_img
            second_img = tmp
            # When we swap the first and second image, then we also need to replace the third by the four image
            # to ensure that the first image have no connection to the third image
            third_img = fourth_img
            label = 0.

        if self.with_negative and 0.3 < torch.rand(1):
            # Negative pair for evaluation
            second_img = third_img
            label = 2.

        if self.transform is not None:
            first_img, second_img = self.transform(first_img, second_img)

        assert isinstance(first_img, torch.Tensor)
        assert isinstance(second_img, torch.Tensor)

        stacked_img = torch.stack([first_img, second_img], dim=0)
        return stacked_img, torch.tensor(label, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.dataset)

