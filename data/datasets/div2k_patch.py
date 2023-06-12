import glob
import logging
import math
import os
import pickle
import random
from enum import Enum
from typing import Callable, Optional, Union

import torch
import torchvision
from PIL import Image
from torchvision.datasets import VisionDataset

from data import transforms

logger = logging.getLogger("pajisaw")
_Target = int


class _Split(Enum):
    TRAIN = "train"
    VAL = "validation"

    @property
    def sub_dir(self) -> str:
        paths = {
            _Split.TRAIN: 'DIV2K_train_HR',  # percentage of the dataset
            _Split.VAL: 'DIV2K_valid_HR',
        }
        return paths[self]

    @staticmethod
    def from_string(name):
        for key in _Split:
            if key.value == name:
                return key


class DIV2KPatch(VisionDataset):
    Target = Union[_Target]
    Split = Union[_Split]

    def __init__(
        self,
        root: str,
        split: "DIV2KPatch.Split",
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        image_size=224,
        erosion_ratio=0.07,
        with_negative=False
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)
        self._split = split
        self.data_dir = os.path.join(root, split.sub_dir)
        entry_file = os.path.join(root, split.sub_dir, 'data.pkl')
        with open(entry_file, 'rb') as f:
            data = pickle.load(f)
        self.entries = data['all_entries']

        self.image_size = image_size
        self.with_negative = with_negative
        self.erosion_ratio = erosion_ratio

        self.cropper_class = torchvision.transforms.RandomCrop
        if split != _Split.TRAIN:
            self.cropper_class = torchvision.transforms.CenterCrop

    @property
    def split(self) -> "DIV2KPatch.Split":
        return self._split

    def load_entry(self, entry):
        with Image.open(os.path.join(self.data_dir, entry['img'])) as f:
            image = f.convert('RGB')

        gap = int(self.image_size * self.erosion_ratio)
        cropper = self.cropper_class((self.image_size - gap * 2, self.image_size - gap * 2))
        return cropper(image)

    def __getitem__(self, index: int):
        entry = self.entries[index]
        first_img = self.load_entry(entry)
        secondary_entry = random.choice(entry['positive'])
        second_img = self.load_entry(secondary_entry)
        if entry['row'] < secondary_entry['row']:
            label = [1., 0., 0., 0.]
        elif entry['col'] < secondary_entry['col']:
            label = [0., 1., 0., 0.]
        elif entry['row'] > secondary_entry['row']:
            label = [0., 0., 1., 0.]
        else:
            label = [0., 0., 0., 1.]

        if self.with_negative and 0.3 > torch.rand(1):
            # Negative pair for evaluation
            secondary_entry = random.choice(entry['negative'])
            second_img = self.load_entry(secondary_entry)

            label = [0., 0., 0., 0.]

        if self.transform is not None:
            first_img, second_img = self.transform(first_img, second_img)

        assert isinstance(first_img, torch.Tensor)
        assert isinstance(second_img, torch.Tensor)

        stacked_img = torch.stack([first_img, second_img], dim=0)
        return stacked_img, torch.tensor(label, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.entries)

