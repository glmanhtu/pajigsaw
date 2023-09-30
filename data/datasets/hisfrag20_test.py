import glob
import logging
import os
from typing import Callable, Optional, Union

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
        samples=None
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)
        self.root_dir = root

        if samples is None:
            samples = glob.glob(os.path.join(root, '**', '*.jpg'), recursive=True)
            samples = sorted(samples)
        self.samples = samples

    def get_group_id(self, index):
        img_path = self.samples[index]
        file_name = os.path.basename(img_path)
        writer_id, page_id, _ = tuple(file_name.split("_"))
        return writer_id

    def __getitem__(self, index: int):
        img_path = self.samples[index]

        with Image.open(img_path) as f:
            image = f.convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(index, dtype=torch.int)

    def __len__(self) -> int:
        return len(self.samples)

