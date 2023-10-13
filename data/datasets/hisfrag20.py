import glob
import logging
import os
import random
from enum import Enum
from typing import Callable, Optional, Union

import albumentations as A
import numpy as np
import torch
import torchvision
from PIL import Image
from torchvision.datasets import VisionDataset

logger = logging.getLogger("pajisaw")
_Target = int


class _Split(Enum):
    TRAIN = "train"
    VAL = "validation"

    def is_train(self):
        return self.value == 'train'

    @staticmethod
    def from_string(name):
        for key in _Split:
            if key.value == name:
                return key


class HisFrag20(VisionDataset):
    Target = Union[_Target]
    Split = Union[_Split]

    def __init__(
        self,
        root: str,
        split: "HisFrag20.Split",
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        image_size=512,
        repeat=1,
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)
        self._split = split
        self.root_dir = root
        self.image_size = image_size
        self.repeat = repeat
        writer_map = {}
        for img in glob.iglob(os.path.join(self.root_dir, 'train', '**', '*.jpg'), recursive=True):
            file_name = os.path.splitext(os.path.basename(img))[0]
            writer_id, page_id, fragment_id = tuple(file_name.split("_"))
            if writer_id not in writer_map:
                writer_map[writer_id] = {}
            if page_id not in writer_map[writer_id]:
                writer_map[writer_id][page_id] = []
            writer_map[writer_id][page_id].append(img)

        writers = sorted(writer_map.keys())
        n_train = int(0.85 * len(writers))
        if split.is_train():
            writers = writers[:n_train]
        else:
            writers = writers[n_train:]
        writer_set = set(writers)
        for writer in list(writer_map.keys()):
            if writer not in writer_set:
                del writer_map[writer]
        self.writers = list(writer_map.keys())
        self.writer_pages = {}
        for writer in writer_map:
            if writer not in self.writer_pages:
                self.writer_pages[writer] = []
            self.writer_pages[writer] += list(writer_map[writer].keys())
        self.writer_map = writer_map

    @property
    def split(self) -> "HisFrag20.Split":
        return self._split

    def __getitem__(self, index: int):
        if index >= len(self.writers):
            index = index % len(self.writers)

        writer_id = self.writers[index]
        page_id = random.choice(self.writer_pages[writer_id])
        img_path = random.choice(self.writer_map[writer_id][page_id])

        with Image.open(img_path) as f:
            first_img = f.convert('RGB')

        if 0.4 > torch.rand(1):
            writer_id_2 = writer_id
            label = 1
        else:
            writer_id_2 = writer_id
            while writer_id_2 == writer_id:
                writer_id_2 = random.choice(self.writers)
            label = 0

        img_path_2 = img_path
        while img_path_2 == img_path:
            page_id_2 = random.choice(self.writer_pages[writer_id_2])
            img_path_2 = random.choice(self.writer_map[writer_id_2][page_id_2])

        with Image.open(img_path_2) as f:
            second_img = f.convert('RGB')

        if self.split.is_train():
            img_transforms = torchvision.transforms.Compose([
                torchvision.transforms.Resize(int(self.image_size * 1.2)),
                torchvision.transforms.RandomResizedCrop(self.image_size, scale=(0.8, 1.2)),
                torchvision.transforms.RandomAffine(5, translate=(0.1, 0.1)),
                torchvision.transforms.RandomGrayscale(p=0.3),
                torchvision.transforms.RandomApply([
                    torchvision.transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                ], p=0.5),
            ])
        else:
            img_transforms = torchvision.transforms.Compose([
                torchvision.transforms.Resize(int(self.image_size * 1.2)),
                torchvision.transforms.CenterCrop(self.image_size)
            ])

        first_img = img_transforms(first_img)
        second_img = img_transforms(second_img)

        if 0.5 > torch.rand(1) and self.split.is_train():
            first_img, second_img = second_img, first_img

        if self.transform is not None:
            first_img, second_img = self.transform(first_img, second_img)

        assert isinstance(first_img, torch.Tensor)
        assert isinstance(second_img, torch.Tensor)

        stacked_img = torch.stack([first_img, second_img], dim=0)
        return stacked_img, torch.tensor([label], dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.writers) * self.repeat

