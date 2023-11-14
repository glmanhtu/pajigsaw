import glob
import logging
import os
import random
from enum import Enum
from typing import Callable, Optional, Union

import albumentations as A
import numpy as np
import torch
import torchvision
from PIL import Image
from torchvision.datasets import VisionDataset

from misc.utils import chunks

logger = logging.getLogger("pajisaw")
_Target = int


class _Split(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"

    @property
    def length(self) -> float:
        split_lengths = {
            _Split.TRAIN: 0.985,  # percentage of the dataset
            _Split.VAL: 0.015
        }
        return split_lengths[self]

    def is_train(self):
        return self.value == 'train'

    def is_val(self):
        return self.value == 'val'

    @staticmethod
    def from_string(name):
        for key in _Split:
            if key.value == name:
                return key


def get_writers(root_dir, proportion=(0., 1.)):
    writer_map = {}
    for img in glob.iglob(os.path.join(root_dir, '**', '*.jpg'), recursive=True):
        file_name = os.path.splitext(os.path.basename(img))[0]
        writer_id, page_id, fragment_id = tuple(file_name.split("_"))
        if writer_id not in writer_map:
            writer_map[writer_id] = {}
        if page_id not in writer_map[writer_id]:
            writer_map[writer_id][page_id] = []
        writer_map[writer_id][page_id].append(img)

    writers = sorted(writer_map.keys())
    n_writers = len(writers)
    from_idx, to_idx = int(proportion[0] * n_writers), int(proportion[1] * n_writers)
    writers = writers[from_idx:to_idx]
    writer_set = set(writers)
    for writer in sorted(writer_map.keys()):
        if writer not in writer_set:
            del writer_map[writer]
    return writers, writer_map


class HisFrag20(VisionDataset):
    Target = Union[_Target]
    Split = Union[_Split]

    def __init__(
        self,
        root: str,
        split: "HisFrag20.Split",
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)
        self._split = split
        self.root_dir = os.path.join(root, split.value)
        self.image_size = 512   # For the Hisfrag dataset, we fixed the image size of 512 pixels
        if not split.is_train():
            raise Exception("This class can only be used for training mode!")

        writers, writer_map = get_writers(self.root_dir, (0., split.length))
        samples = []
        for writer in sorted(writer_map.keys()):
            patches = sorted([x for page in writer_map[writer] for x in writer_map[writer][page]])
            samples += chunks(patches, 5)
        self.writer_map = writer_map
        self.samples = samples
        self.writers = writers

    @property
    def split(self) -> "HisFrag20.Split":
        return self._split

    def __getitem__(self, index: int):
        img_path = random.choice(self.samples[index])
        file_name = os.path.splitext(os.path.basename(img_path))[0]
        writer_id, page_id, fragment_id = tuple(file_name.split("_"))

        with Image.open(img_path) as f:
            first_img = f.convert('RGB')

        if 0.4 > torch.rand(1):
            writer_id_2 = writer_id
            label = 1
        else:
            writer_id_2 = writer_id
            while writer_id_2 == writer_id:
                writer_id_2 = random.choice(self.writers)
            label = 0

        img_path_2 = img_path
        while img_path_2 == img_path:
            page_id_2 = random.choice(list(self.writer_map[writer_id_2].keys()))
            img_path_2 = random.choice(self.writer_map[writer_id_2][page_id_2])

        with Image.open(img_path_2) as f:
            second_img = f.convert('RGB')

        if self.split.is_train():
            train_transform = A.Compose(
                [
                    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=10, p=0.5),
                ]
            )

            img_transforms = torchvision.transforms.Compose([
                torchvision.transforms.RandomCrop(self.image_size, pad_if_needed=True),
                lambda x: np.array(x),
                lambda x: train_transform(image=x)['image'],
                torchvision.transforms.ToPILImage(),
                torchvision.transforms.RandomApply([
                    torchvision.transforms.GaussianBlur((3, 3), (1.0, 2.0)),
                    torchvision.transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                ], p=0.5)
            ])
        else:
            img_transforms = torchvision.transforms.Compose([
                torchvision.transforms.CenterCrop(self.image_size)
            ])

        first_img = img_transforms(first_img)
        second_img = img_transforms(second_img)

        if self.transform is not None:
            first_img, second_img = self.transform(first_img, second_img)

        assert isinstance(first_img, torch.Tensor)
        assert isinstance(second_img, torch.Tensor)

        stacked_img = torch.stack([first_img, second_img], dim=0)
        return stacked_img, torch.tensor([label], dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.samples)


class HisFrag20Test(VisionDataset):
    Target = Union[_Target]
    Split = Union[_Split]

    def __init__(
        self,
        root: str,
        split: "HisFrag20Test.Split",
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        samples = None,
        lower_bound = 0
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)
        if split.is_train():
            raise Exception('This class can only be used in Validation or Testing mode!')

        if samples is None:
            sub_dir = _Split.TRAIN.value  # Train and Val use the same training set
            if split is _Split.TEST:
                sub_dir = split.value
            root_dir = os.path.join(root, sub_dir)
            proportion = 0., 1.     # Testing mode uses all samples
            if split.is_val():
                proportion = 1. - split.length, 1.
            writers, writer_map = get_writers(root_dir, proportion)

            samples = []
            for writer_id in writers:
                for page_id in writer_map[writer_id]:
                    samples += writer_map[writer_id][page_id]

            samples = sorted(samples)

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


class HisFrag20GT(VisionDataset):
    Target = Union[_Target]
    Split = _Split

    def __init__(
        self,
        root: str,
        split: "HisFrag20GT.Split",
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)
        self.root_dir = root
        sub_dir = _Split.TRAIN.value  # Train and Val use the same training set
        self.root_dir = os.path.join(root, sub_dir)
        proportion = 1. - split.length, 1.
        writers, writer_map = get_writers(self.root_dir, proportion)

        samples = []
        for writer_id in writers:
            for page_id in writer_map[writer_id]:
                samples += writer_map[writer_id][page_id]

        self.samples = sorted(samples)
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
