import glob
import math
import os
from enum import Enum
from typing import Union, Optional, Callable

import imagesize
from PIL import Image
from ml_engine.data.grouping import add_items_to_group
from torch.utils.data import Dataset
from torchvision.datasets import VisionDataset


class _Split(Enum):
    TRAIN = "train"
    VAL = "validation"
    ALL = "all"

    @property
    def length(self) -> float:
        split_lengths = {
            _Split.TRAIN: 0.85,  # percentage of the dataset
            _Split.VAL: 0.15,
            _Split.ALL: 1.
        }
        return split_lengths[self]

    def is_train(self):
        return self.value == 'train'

    def is_val(self):
        return self.value == 'validation'

    @staticmethod
    def from_string(name):
        for key in _Split:
            if key.value == name:
                return key


class MichiganDataset(Dataset):
    Split = Union[_Split]

    def __init__(self, dataset_path: str, split: "MichiganDataset.Split", transforms,
                 samples=None, val_n_items_per_writer=None):
        self.dataset_path = dataset_path
        self.samples = samples
        if samples is None:
            files = glob.glob(os.path.join(dataset_path, '**', '*.png'), recursive=True)
            files.extend(glob.glob(os.path.join(dataset_path, '**', '*.jpg'), recursive=True))

            image_map = {}
            groups = []
            for file in files:
                file_name_components = file.split(os.sep)
                im_name, rv, sum_det, sub_name, im_type, _, _ = file_name_components[-7:]
                add_items_to_group([im_name, sub_name], groups)
                if rv != 'front':
                    continue
                if im_type != 'papyrus':
                    continue
                image_map.setdefault(im_name, {}).setdefault(sum_det, []).append(file)

            self.fragment_to_group = {}
            self.fragment_to_group_id = {}
            self.groups = groups

            for idx, group in enumerate(groups):
                for fragment in group:
                    self.fragment_to_group_id[fragment] = idx
                    for fragment2 in group:
                        self.fragment_to_group.setdefault(fragment, set([])).add(fragment2)

            images = {}
            for img in image_map:
                images[img] = []
                key = 'detail'
                if key not in image_map[img]:
                    key = 'summary'
                images[img] = image_map[img][key]
                if val_n_items_per_writer is not None and split.is_val():
                    images[img] = images[img][:val_n_items_per_writer]

            self.image_names = sorted(images.keys())

            if split == MichiganDataset.Split.TRAIN:
                self.image_names = self.image_names[: int(len(self.image_names) * split.length)]
            elif split == MichiganDataset.Split.VAL:
                self.image_names = self.image_names[-int(len(self.image_names) * split.length):]
            else:
                self.image_names = self.image_names

            self.image_idxes = {k: i for i, k in enumerate(self.image_names)}
            self.data = []
            self.data_labels = []
            for img in self.image_names:
                data, labels = [], []
                for fragment in sorted(images[img]):
                    data.append(fragment)
                    labels.append(self.fragment_to_group_id[img])

                if split.is_val() and len(data) < 2:
                    continue

                self.data.extend(data)
                self.data_labels.extend(labels)
        else:
            self.data = samples

        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        fragment = self.data[idx]

        with Image.open(fragment) as img:
            image = self.transforms(img.convert('RGB'))

        label = self.data_labels[idx]
        return image, label


class MichiganTest(MichiganDataset):
    Split = Union[_Split]

    def __init__(self, dataset_path: str, split: "MichiganDataset.Split", transforms, lower_bound=0,
                 samples = None,
                 val_n_items_per_writer=2):
        super().__init__(dataset_path, split, transforms, samples=samples, val_n_items_per_writer=val_n_items_per_writer)
        self.lower_bound = lower_bound

    def __getitem__(self, index: int):
        index = index + self.lower_bound

        fragment = self.data[index]

        with Image.open(fragment) as img:
            image = self.transforms(img.convert('RGB'))

        return image, index

    def __len__(self) -> int:
        return len(self.data) - self.lower_bound
