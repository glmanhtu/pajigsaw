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
        max_n_authors: int = None,
        samples = None,
        lower_bound = 0
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)
        self.root_dir = root
        if samples is None:
            writer_map = {}
            for img in glob.iglob(os.path.join(self.root_dir, 'test', '**', '*.jpg'), recursive=True):
                file_name = os.path.splitext(os.path.basename(img))[0]
                writer_id, page_id, fragment_id = tuple(file_name.split("_"))
                if writer_id not in writer_map:
                    writer_map[writer_id] = {}
                if page_id not in writer_map[writer_id]:
                    writer_map[writer_id][page_id] = []
                writer_map[writer_id][page_id].append(img)

            writers = sorted(writer_map.keys())
            if max_n_authors is not None:
                writers = writers[:max_n_authors]

            samples = []
            for writer_id in writers:
                for page_id in writer_map[writer_id]:
                    samples += writer_map[writer_id][page_id]

        self.samples = samples
        self.lower_bound = lower_bound

    def __getitem__(self, index: int):
        index = index + self.lower_bound
        img_path = self.samples[index]

        with Image.open(img_path) as f:
            image = f.convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, index

    def __len__(self) -> int:
        return len(self.samples) - self.lower_bound


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

        return image, self.pairs[index].type(torch.float16), x1_id

    def __len__(self) -> int:
        return len(self.pairs)


class HisFrag20GT(VisionDataset):
    Target = Union[_Target]

    def __init__(
        self,
        root: str,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        max_n_authors: int = None,
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)
        self.root_dir = root
        writer_map = {}
        for img in glob.iglob(os.path.join(self.root_dir, 'test', '**', '*.jpg'), recursive=True):
            file_name = os.path.splitext(os.path.basename(img))[0]
            writer_id, page_id, fragment_id = tuple(file_name.split("_"))
            if writer_id not in writer_map:
                writer_map[writer_id] = {}
            if page_id not in writer_map[writer_id]:
                writer_map[writer_id][page_id] = []
            writer_map[writer_id][page_id].append(img)

        writers = sorted(writer_map.keys())
        if max_n_authors is not None:
            writers = writers[:max_n_authors]

        samples = []
        for writer_id in writers:
            for page_id in writer_map[writer_id]:
                samples += writer_map[writer_id][page_id]

        self.samples = samples
        indicates = torch.arange(len(samples)).type(torch.int)
        pairs = torch.combinations(indicates, r=2, with_replacement=True)
        self.pairs = pairs

    def __getitem__(self, index: int):
        x1_id, x2_id = tuple(self.pairs[index])
        img_path = self.samples[x1_id.item()]

        with Image.open(img_path) as f:
            image = f.convert('RGB')

        img2_path = self.samples[x2_id.item()]
        with Image.open(img2_path) as f:
            image2 = f.convert('RGB')

        if self.transform:
            image = self.transform(image)
            image2 = self.transform(image2)
        stacked_img = torch.stack([image, image2], dim=0)
        return stacked_img, self.pairs[index]

    def __len__(self) -> int:
        return len(self.pairs)
