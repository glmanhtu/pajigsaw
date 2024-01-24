import glob
import logging
import math
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
            _Split.TRAIN: 0.93,  # percentage of the dataset
            _Split.VAL: 0.07,
            _Split.TEST: 1.
        }
        return split_lengths[self]

    @property
    def sub_dir(self):
        dirs = {
            _Split.TRAIN: "train",
            _Split.VAL: "train",
            _Split.TEST: "test"
        }
        return dirs[self]

    def is_train(self):
        return self.value == 'train'

    def is_val(self):
        return self.value == 'val'

    def is_test(self):
        return self.value == 'test'

    @staticmethod
    def from_string(name):
        for key in _Split:
            if key.value == name:
                return key


def get_writers(root_dir, proportion=(0., 1.)):
    writer_map = {}
    for img in sorted(glob.glob(os.path.join(root_dir, '**', '*.jpg'), recursive=True)):
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
    for writer in list(writer_map.keys()):
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
        self.root_dir = os.path.join(root, split.sub_dir)

        proportion = 0., split.length
        if split.is_val():
            proportion = 1. - split.length, 1.
        writers, writer_map = get_writers(self.root_dir, proportion)

        self.writer_to_idx = {x: i for i, x in enumerate(writers)}
        samples, labels = [], []
        for writer in sorted(writer_map.keys()):
            for page in sorted(writer_map[writer].keys()):
                samples += writer_map[writer][page]
                labels += [self.writer_to_idx[writer]] * len(writer_map[writer][page])
        self.writer_map = writer_map
        self.data_labels = labels
        self.samples = samples
        self.writers = writers

    @property
    def split(self) -> "HisFrag20.Split":
        return self._split

    def __getitem__(self, index: int):
        img_path = self.samples[index]
        file_name = os.path.splitext(os.path.basename(img_path))[0]
        writer_id, page_id, fragment_id = tuple(file_name.split("_"))

        with Image.open(img_path) as f:
            image = f.convert('RGB')

        label = self.writer_to_idx[writer_id]
        if self.transform is not None:
            image = self.transform(image)

        assert isinstance(image, torch.Tensor)

        return image, label

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
        lower_bound = 0,
        val_n_items_per_writer=2,
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)
        if split.is_train():
            raise Exception('This class can only be used in Validation or Testing mode!')

        if samples is None:
            root_dir = os.path.join(root, split.sub_dir)
            proportion = 0., 1.     # Testing mode uses all samples
            if split.is_val():
                proportion = 1. - split.length, 1.
            writers, writer_map = get_writers(root_dir, proportion)

            samples = []
            for writer_id in writers:
                page_patches = []
                for page_id in sorted(writer_map[writer_id].keys()):
                    page_patches += sorted(writer_map[writer_id][page_id])

                if split.is_val():
                    n_items_per_chunk = math.ceil(len(page_patches) / val_n_items_per_writer)
                    page_patches = chunks(page_patches, n_items_per_chunk)[0]

                samples += page_patches
            samples = samples

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
        val_n_items_per_writer=2,
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)
        self.root_dir = root
        sub_dir = _Split.TRAIN.value  # Train and Val use the same training set
        self.root_dir = os.path.join(root, sub_dir)
        proportion = 1. - split.length, 1.
        writers, writer_map = get_writers(self.root_dir, proportion)

        samples = []
        for writer_id in writers:
            page_patches = []
            for page_id in sorted(writer_map[writer_id].keys()):
                page_patches += sorted(writer_map[writer_id][page_id])

            if split.is_val():
                n_items_per_chunk = math.ceil(len(page_patches) / val_n_items_per_writer)
                page_patches = chunks(page_patches, n_items_per_chunk)[0]

            samples += page_patches

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
