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

from misc.utils import chunks

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
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)
        self._split = split
        self.root_dir = root
        self.image_size = 512
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
        samples = []
        for writer in sorted(writer_map.keys()):
            if writer not in writer_set:
                del writer_map[writer]
            else:
                patches = sorted([x for page in writer_map[writer] for x in writer_map[writer][page]])
                samples += chunks(patches, 5)
        self.writer_map = writer_map
        self.samples = samples
        self.writers = sorted(writer_set)

    @property
    def split(self) -> "HisFrag20.Split":
        return self._split

    def __getitem__(self, index: int):
        img_path = random.choice(self.samples[index])
        file_name = os.path.splitext(os.path.basename(img_path))[0]
        writer_id, page_id, fragment_id = tuple(file_name.split("_"))

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
            page_id_2 = random.choice(list(self.writer_map[writer_id_2].keys()))
            img_path_2 = random.choice(self.writer_map[writer_id_2][page_id_2])

        with Image.open(img_path_2) as f:
            second_img = f.convert('RGB')

        if self.split.is_train():
            train_transform = A.Compose(
                [
                    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=10, p=0.5),
                ]
            )

            img_transforms = torchvision.transforms.Compose([
                torchvision.transforms.RandomCrop(self.image_size, pad_if_needed=True),
                lambda x: np.array(x),
                lambda x: train_transform(image=x)['image'],
                torchvision.transforms.ToPILImage(),
                torchvision.transforms.RandomApply([
                    torchvision.transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                ], p=0.5)
            ])
        else:
            img_transforms = torchvision.transforms.Compose([
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
        return len(self.samples)
