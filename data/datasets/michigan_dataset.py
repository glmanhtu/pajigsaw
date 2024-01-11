import glob
import math
import os
from enum import Enum
from typing import Union, Optional, Callable

import imagesize
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import VisionDataset


class _Split(Enum):
    TRAIN = "train"
    VAL = "validation"

    @property
    def length(self) -> float:
        split_lengths = {
            _Split.TRAIN: 0.85,  # percentage of the dataset
            _Split.VAL: 0.15
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
                 min_size=112, samples=None):
        self.dataset_path = dataset_path
        if samples is None:
            files = glob.glob(os.path.join(dataset_path, '**', '*.png'), recursive=True)
            files.extend(glob.glob(os.path.join(dataset_path, '**', '*.jpg'), recursive=True))
            image_map = {}
            for file in files:
                file_name_components = file.split(os.sep)
                im_name, rv, sum_det, _, im_type, _, _ = file_name_components[-7:]
                if rv != 'front':
                    continue
                if im_type != 'papyrus':
                    continue
                image_map.setdefault(im_name, {}).setdefault(sum_det, []).append(file)

            images = {}
            for img in image_map:
                key = 'detail'
                if key not in image_map[img]:
                    key = 'summary'
                images[img] = image_map[img][key]

            self.labels = sorted(images.keys())
            self.__label_idxes = {k: i for i, k in enumerate(self.labels)}

            if split == MichiganDataset.Split.TRAIN:
                self.labels = self.labels[: int(len(self.labels) * split.length)]
            else:
                self.labels = self.labels[-int(len(self.labels) * split.length):]

            self.data = []
            self.data_labels = []
            for img in self.labels:
                data, labels = [], []
                for fragment in sorted(images[img]):
                    width, height = imagesize.get(fragment)
                    if width * height < min_size * min_size:
                        continue

                    # ratio = max(round((width * height) / (im_size * im_size)), 1) if split.is_train() else 1
                    # for _ in range(int(ratio)):
                    data.append(fragment)
                    labels.append(self.__label_idxes[img])

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
        super().__init__(dataset_path, split, transforms, samples=samples)
        self.lower_bound = lower_bound

    def __getitem__(self, index: int):
        index = index + self.lower_bound

        fragment = self.data[index]

        with Image.open(fragment) as img:
            image = self.transforms(img.convert('RGB'))

        return image, index

    def __len__(self) -> int:
        return len(self.data) - self.lower_bound
