import glob
import logging
import math
import os
import random
from enum import Enum
from typing import Callable, Optional, Union

import albumentations as A
import imagesize
import numpy as np
import torch
import torchvision
from PIL import Image
from torchvision.datasets import VisionDataset

from data import transforms
from data.transforms import CustomRandomCrop, make_square
from data.utils import UnableToCrop

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
        self.resizer = torchvision.transforms.Compose([
            lambda x: make_square(x),
            torchvision.transforms.Resize(size=self.image_size)
        ])
        self.repeat = repeat
        writer_map = {}
        samples = []
        for img in glob.iglob(os.path.join(self.root_dir, '**', '*.jpg'), recursive=True):
            file_name = os.path.splitext(os.path.basename(img))[0]
            writer_id, page_id, fragment_id = tuple(file_name.split("_"))
            if writer_id not in writer_map:
                writer_map[writer_id] = {}
            if page_id not in writer_map[writer_id]:
                writer_map[writer_id][page_id] = []
            writer_map[writer_id][page_id].append(img)
            samples.append(img)
        self.writer_map = writer_map
        self.samples = samples

    @property
    def split(self) -> "HisFrag20.Split":
        return self._split

    def __getitem__(self, index: int):
        if index >= len(self.samples):
            index = index % len(self.samples)

        img_path = self.samples[index]
        file_name = os.path.splitext(os.path.basename(img_path))[0]
        writer_id, page_id, _ = tuple(file_name.split("_"))

        with Image.open(img_path) as f:
            image = f.convert('RGB')

        first_img = self.resizer(image)

        if 0.5 > torch.rand(1):
            writer_id_2 = writer_id
            label = 1
        else:
            writer_id_2 = writer_id
            while writer_id_2 == writer_id:
                writer_id_2 = random.choice(list(self.writer_map.keys()))
            label = 0

        img_path_2 = img_path
        while img_path_2 == img_path:
            page_id_2 = random.choice(list(self.writer_map[writer_id_2].keys()))
            img_path_2 = random.choice(list(self.writer_map[writer_id_2][page_id_2]))

        with Image.open(img_path_2) as f:
            image = f.convert('RGB')
        second_img = self.resizer(image)

        if self.split.is_train():
            train_transform = A.Compose(
                [
                    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.15, rotate_limit=20, p=0.5),
                    A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
                ]
            )
            img_transforms = torchvision.transforms.Compose([
                # torchvision.transforms.RandomHorizontalFlip(),
                # torchvision.transforms.RandomVerticalFlip(),
                lambda x: np.array(x),
                lambda x: train_transform(image=x)['image'],
                torchvision.transforms.ToPILImage(),
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
        return len(self.samples) * self.repeat

