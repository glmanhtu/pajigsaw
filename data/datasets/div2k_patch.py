import glob
import logging
import os
import random
from enum import Enum
from typing import Callable, Optional, Union

import torchvision
from PIL import Image

from data.datasets.imnet_patch import ImNetPatch

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

    def is_train(self):
        return self.value == 'train'

    @staticmethod
    def from_string(name):
        for key in _Split:
            if key.value == name:
                return key


class DIV2KPatch(ImNetPatch):
    Target = Union[_Target]
    Split = Union[_Split]

    def __init__(
        self,
        root: str,
        split: "DIV2KPatch.Split",
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        image_size=64,
        erosion_ratio=0.07,
        with_negative=False,
        repeat=5
    ) -> None:
        super().__init__(root, split, transforms, transform, target_transform, image_size, erosion_ratio, with_negative)
        self.repeat = repeat

    @property
    def split(self) -> "DIV2KPatch.Split":
        return self._split

    def load_dataset(self):
        dataset_dir = os.path.join(self.root_dir, self.split.sub_dir)
        images = []
        for root, dirs, files in os.walk(dataset_dir):
            for file in files:
                if file.lower().endswith((".jpg", ".png")):
                    images.append(os.path.join(root, file))
        return images

    def read_image(self, index):
        if index >= len(self.dataset):
            index = index % len(self.dataset)
        img_path = self.dataset[index]
        with Image.open(img_path) as f:
            image = f.convert('RGB')

        if self.split.is_train():
            scale = random.uniform(0.8, 1.2)
            transforms = torchvision.transforms.Compose([
                torchvision.transforms.RandomApply(
                    [torchvision.transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2)],
                    p=0.8
                ),
                torchvision.transforms.Resize((int(scale * image.height), int(scale * image.width)))
            ])

            image = transforms(image)

        return image

    def __len__(self) -> int:
        if self.split.is_train():
            return len(self.dataset) * (self.repeat + 1)
        else:
            return len(self.dataset)



