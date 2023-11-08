import json
import logging
import os
from enum import Enum
from typing import Callable, Optional, Union

from torchvision.datasets import VisionDataset

logger = logging.getLogger("pajisaw")
_Target = int


class _Split(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"

    def is_train(self):
        return self.value == 'train'

    @staticmethod
    def from_string(name):
        for key in _Split:
            if key.value == name:
                return key


class Pajigsaw(VisionDataset):
    Target = Union[_Target]
    Split = Union[_Split]

    def __init__(
        self,
        root: str,
        split: "Pajigsaw.Split",
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        image_size=512,
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)
        with open(os.path.join(root, f'{split.value}.json')) as f:
            dataset = json.load(f)
        records = {}
        for img_name in dataset:
            records[img_name] = []
            for fragment in dataset[img_name]['Fragment1v1Rotate90']:
                if fragment['degree'] == 0:
                    records[img_name].append(fragment)
        self.samples = records
        self._split = split
        self.root_dir = root
    @property
    def split(self) -> "Pajigsaw.Split":
        return self._split

    def __len__(self) -> int:
        return len(self.dataset)



