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
        erosion_ratio=0.07,
        with_negative=False
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)
        self._split = split
        self.root_dir = root

        self.image_size = image_size
        self.with_negative = with_negative
        self.erosion_ratio = erosion_ratio

        self.cropper_class = torchvision.transforms.RandomCrop
        if not split.is_train():
            self.cropper_class = torchvision.transforms.CenterCrop
        os.makedirs(self.root_dir, exist_ok=True)
        print('Loading imagenet...')
        self.dataset = load_dataset("imagenet-1k", split=self._split.value, cache_dir=self.root_dir)
        categories_path = os.path.join(root, 'categories.json')
        if os.path.isfile(categories_path):
            with open(categories_path) as f:
                categories = json.load(f)
        else:
            categories = {}
            for idx, item in enumerate(self.dataset):
                categories.setdefault(item['label'], []).append(idx)
            with open(categories_path, 'w') as f:
                json.dump(categories, f)
        self.categories_map = categories
        self.categories = sorted(categories.keys())

    @property
    def split(self) -> "ImNetPatch.Split":
        return self._split

    def __getitem__(self, index: int):
        item = self.dataset[index]
        first_img = item['image'].convert('RGB')
        category = item['label']
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(int(self.image_size * 1.2)),
            self.cropper_class(self.image_size)
        ])

        category2 = category
        if 0.5 > torch.rand(1):
            label = 1.
        else:
            while category2 == category:
                category2 = random.choice(self.categories)
            label = 0.

        item2 = self.dataset[random.choice(self.categories_map[category2])]
        second_img = item2['image'].convert('RGB')

        first_img = transform(first_img)
        second_img = transform(second_img)
        if self.transform is not None:
            first_img, second_img = self.transform(first_img, second_img)

        assert isinstance(first_img, torch.Tensor)
        assert isinstance(second_img, torch.Tensor)

        stacked_img = torch.stack([first_img, second_img], dim=0)
        return stacked_img, torch.tensor([label], dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.dataset)

