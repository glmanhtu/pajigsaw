import json
import logging
import math
import os
import random
from enum import Enum
from typing import Callable, Optional, Union

import torch
import torchvision
from datasets import load_dataset
from torchvision.datasets import VisionDataset

from data import transforms
from data.transforms import RandomResize

logger = logging.getLogger("pajisaw")
_Target = int


class _Split(Enum):
    TRAIN = "train"
    VAL = "validation"
    TEST = "test"  # NOTE: torchvision does not support the test split

    def is_train(self):
        return self.value == 'train'

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
        image_size=512,
        erosion_ratio=0.07
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)
        self._split = split
        self.root_dir = root

        self.image_size = image_size
        self.erosion_ratio = erosion_ratio
        self.cropper_class = torchvision.transforms.RandomCrop
        if not split.is_train():
            self.cropper_class = torchvision.transforms.CenterCrop
        os.makedirs(self.root_dir, exist_ok=True)
        self.dataset = self.get_dataset(split=self._split.value, cache_dir=self.root_dir)
        categories_path = os.path.join(root, f'{split.value}_categories.json')
        if os.path.isfile(categories_path):
            with open(categories_path) as f:
                categories = json.load(f)
        else:
            categories = {}
            for idx, item in enumerate(self.dataset):
                _, category = self.extract_item(item)
                categories.setdefault(category, []).append(idx)
            with open(categories_path, 'w') as f:
                json.dump(categories, f)
        self.categories_map = categories
        self.categories = sorted(categories.keys())

    def extract_item(self, item):
        return item['image'], str(item['label'])

    def get_dataset(self, split, cache_dir):
        return load_dataset("imagenet-1k", split=split, cache_dir=cache_dir)

    @property
    def split(self) -> "ImNetPatch.Split":
        return self._split

    def __getitem__(self, index: int):
        image, first_category = self.extract_item(self.dataset[index])
        image = image.convert('RGB')

        # Crop the image into a grid of 3 x 2 patches
        crops = transforms.crop(image, 2, 2)
        erosion_ratio = self.erosion_ratio
        if self._split.is_train():
            erosion_ratio = random.uniform(self.erosion_ratio, self.erosion_ratio * 2)
        piece_size_erosion = math.ceil(crops[0].width * (1 - erosion_ratio))
        cropper = torchvision.transforms.CenterCrop(piece_size_erosion)
        first_img = cropper(crops[0])

        # Second image is next to the first image
        second_img = cropper(crops[1])

        # Third image is right below the second image
        third_img = cropper(crops[3])

        # Fourth mage is right below the first image
        fourth_img = cropper(crops[2])

        label = [1., 0., 0., 0.]
        if 0.3 > torch.rand(1):
            if 0.5 < torch.rand(1):
                second_img, third_img = third_img, second_img
            else:
                second_img = cropper(crops[2])

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

        if self.split.is_train():
            img_transforms = torchvision.transforms.Compose([
                RandomResize(self.image_size, ratio=(1.0, 1.4)),
                torchvision.transforms.RandomCrop(self.image_size, pad_if_needed=True)
            ])
        else:
            img_transforms = torchvision.transforms.Compose([
                torchvision.transforms.Resize(int(self.image_size * 1.2)),
                torchvision.transforms.CenterCrop(self.image_size)
            ])

        first_img = img_transforms(first_img)
        second_img = img_transforms(second_img)

        if self.transform is not None:
            first_img, second_img = self.transform(first_img, second_img)

        assert isinstance(first_img, torch.Tensor)
        assert isinstance(second_img, torch.Tensor)

        stacked_img = torch.stack([first_img, second_img], dim=0)
        return stacked_img, torch.tensor(label, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.dataset)

