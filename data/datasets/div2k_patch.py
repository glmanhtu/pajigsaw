import glob
import logging
import os
from enum import Enum
from typing import Callable, Optional, Union

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
        with_negative=False
    ) -> None:
        super().__init__(root, split, transforms, transform, target_transform, image_size, erosion_ratio, with_negative)

    @property
    def split(self) -> "DIV2KPatch.Split":
        return self._split

    def load_dataset(self):
        dataset_dir = os.path.join(self.root_dir, self.split.sub_dir)
        return sorted(glob.glob(os.path.join(dataset_dir, '**', '*.jpg'), recursive=True))

    def read_image(self, index):
        img_path = self.dataset[index]
        with Image.open(img_path) as f:
            image = f.convert('RGB')
        return image



