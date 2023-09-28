import glob
import logging
import os
from typing import Callable, Optional, Union

import torch
import torchvision
from PIL import Image
from torchvision.datasets import VisionDataset

from data.transforms import make_square

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
        image_size=512,
        repeat=1,
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)
        self.root_dir = root
        self.image_size = image_size
        self.resizer = torchvision.transforms.Compose([
            lambda x: make_square(x),
            torchvision.transforms.Resize(size=self.image_size)
        ])
        self.repeat = repeat
        writer_map = {}
        samples = []
        for img in glob.iglob(os.path.join(self.root_dir, 'validation', '**', '*.jpg'), recursive=True):
            file_name = os.path.splitext(os.path.basename(img))[0]
            writer_id, page_id, fragment_id = tuple(file_name.split("_"))
            if writer_id not in writer_map:
                writer_map[writer_id] = {}
            if page_id not in writer_map[writer_id]:
                writer_map[writer_id][page_id] = []
            writer_map[writer_id][page_id].append(img)
            samples.append(img)

        self.samples = samples

        self.writer_map = writer_map

    def get_group_id(self, index):
        img_path = self.samples[index]
        file_name = os.path.basename(img_path)
        writer_id, page_id, _ = tuple(file_name.split("_"))
        return writer_id

    def __getitem__(self, index: int):
        i = int(index / len(self.samples))
        j = index - i * len(self.samples)
        img_path = self.samples[i]

        with Image.open(img_path) as f:
            image = f.convert('RGB')

        first_img = self.resizer(image)

        img_path_2 = self.samples[j]
        with Image.open(img_path_2) as f:
            image = f.convert('RGB')
        second_img = self.resizer(image)

        if self.transform is not None:
            first_img, second_img = self.transform(first_img, second_img)

        assert isinstance(first_img, torch.Tensor)
        assert isinstance(second_img, torch.Tensor)

        stacked_img = torch.stack([first_img, second_img], dim=0)
        return stacked_img, torch.tensor([i, j], dtype=torch.int)

    def __len__(self) -> int:
        return len(self.samples) * len(self.samples)

