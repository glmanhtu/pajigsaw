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
