import glob
import logging
import os
from typing import Callable, Optional, Union
import numpy as np
import torch
from PIL import Image
from torchvision.datasets import VisionDataset

logger = logging.getLogger("pajisaw")
_Target = int


class HisFrag20Test(VisionDataset):
    Target = Union[_Target]

    def __init__(
        self,
        root: str,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        samples = None,
        x1_offset = 0
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)
        self.root_dir = root
        if samples is None:
            samples = glob.glob(os.path.join(root, '**', '*.jpg'), recursive=True)
            samples = sorted(samples)
        self.samples = samples
        self.x1_offset = x1_offset

    def __getitem__(self, index: int):
        img_path = self.samples[index]

        with Image.open(img_path) as f:
            image = f.convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(index + self.x1_offset, dtype=torch.int)

    def __len__(self) -> int:
        return len(self.samples)


class HisFrag20X2(VisionDataset):
    Target = Union[_Target]

    def __init__(
        self,
        root: str,
        samples,
        pairs,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)
        self.root_dir = root
        self.samples = samples
        self.pairs = pairs

    def __getitem__(self, index: int):
        x1_id, x2_id = tuple(self.pairs[index])
        img_path = self.samples[x2_id.item()]

        with Image.open(img_path) as f:
            image = f.convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, self.pairs[index], x1_id

    def __len__(self) -> int:
        return len(self.pairs)