import glob
import logging
import os
from typing import Callable, Optional, Union

import torch
import torchvision
import tqdm
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import VisionDataset
from torchlmdb import LMDBDataset
from data.transforms import make_square

logger = logging.getLogger("pajisaw")
_Target = int


class HisFrag(Dataset):
    def __init__(self, root, image_size):
        self.image_size = image_size
        self.resizer = torchvision.transforms.Compose([
            lambda x: make_square(x),
            torchvision.transforms.Resize(size=self.image_size)
        ])
        samples = glob.glob(os.path.join(root, 'validation', '**', '*.jpg'), recursive=True)
        self.samples = sorted(samples)
        self.transform = None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        img_path = self.samples[index]

        with Image.open(img_path) as f:
            image = f.convert('RGB')

        first_img = self.resizer(image)
        if self.transform:
            first_img = self.transform(first_img)
        return first_img, index


class HisFrag20Test(VisionDataset):
    Target = Union[_Target]

    def __init__(
        self,
        root: str,
        cache_dir: str,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        image_size=512,
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)
        self.root_dir = root
        dataset = HisFrag(root, image_size)
        self.db_lmdb = LMDBDataset(dataset, cache_dir, "hisfrag_test")

        self.items = []
        self.samples = dataset.samples
        for i in tqdm.tqdm(range(len(self.samples))):
            for j in range(i, len(self.samples)):
                self.items.append((i, j))

    def get_group_id(self, index):
        img_path = self.samples[index]
        file_name = os.path.basename(img_path)
        writer_id, page_id, _ = tuple(file_name.split("_"))
        return writer_id

    def __getitem__(self, index: int):
        i, j = self.items[index]

        first_img, i_1 = self.db_lmdb[i]
        assert i == i_1
        second_img, j_1 = self.db_lmdb[j]
        assert j == j_1

        if self.transform is not None:
            first_img, second_img = self.transform(first_img, second_img)

        assert isinstance(first_img, torch.Tensor)
        assert isinstance(second_img, torch.Tensor)

        stacked_img = torch.stack([first_img, second_img], dim=0)
        return stacked_img, torch.tensor([i, j], dtype=torch.int)

    def __len__(self) -> int:
        return len(self.items)

