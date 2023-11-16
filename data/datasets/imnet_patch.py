import json
import logging
import os
import random
from enum import Enum
from typing import Callable, Optional, Union

import torch
import torchvision
from datasets import load_dataset
from torchvision.datasets import VisionDataset

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
        image_size=224,
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)
        self._split = split
        self.root_dir = root

        self.image_size = image_size

        self.cropper_class = torchvision.transforms.RandomCrop
        if not split.is_train():
            self.cropper_class = torchvision.transforms.CenterCrop
        os.makedirs(self.root_dir, exist_ok=True)
        print('Loading imagenet...')
        self.dataset = self.get_dataset(split=self._split.value, cache_dir=self.root_dir)
        categories_path = os.path.join(root, f'{split.value}_categories.json')
        if os.path.isfile(categories_path):
            with open(categories_path) as f:
                categories = json.load(f)
        else:
            categories = {}
            for idx, item in enumerate(self.dataset):
                categories.setdefault(self.extract_item(item)[1], []).append(idx)
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
        first_img, first_category = self.extract_item(self.dataset[index])
        first_img = first_img.convert('RGB')

        second_category = first_category
        if 0.5 > torch.rand(1):
            label = 1.
        else:
            while second_category == first_category:
                second_category = random.choice(self.categories)
            label = 0.

        second_index = random.choice(self.categories_map[str(second_category)])
        second_img, second_label = self.extract_item(self.dataset[second_index])
        assert second_label == second_category, f"Incorrect labeling, {second_category} vs {second_label}"
        second_img = second_img.convert('RGB')

        if self.split.is_train():
            img_transforms = torchvision.transforms.Compose([
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.RandomVerticalFlip(),
                RandomResize(self.image_size, ratio=(0.6, 1.0)),
                torchvision.transforms.RandomCrop(self.image_size, pad_if_needed=True)
            ])
        else:
            img_transforms = torchvision.transforms.Compose([
                torchvision.transforms.Resize(int(self.image_size * 0.8)),
                torchvision.transforms.CenterCrop(self.image_size)
            ])

        first_img = img_transforms(first_img)
        second_img = img_transforms(second_img)
        if self.transform is not None:
            first_img, second_img = self.transform(first_img, second_img)

        assert isinstance(first_img, torch.Tensor)
        assert isinstance(second_img, torch.Tensor)

        stacked_img = torch.stack([first_img, second_img], dim=0)
        return stacked_img, torch.tensor([label], dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.dataset)

