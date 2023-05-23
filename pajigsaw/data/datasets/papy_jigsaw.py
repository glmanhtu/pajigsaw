import glob
import logging
import os
import random
from enum import Enum
from typing import Callable, Optional, Union

import torch
from PIL import Image
from torchvision.datasets import VisionDataset

logger = logging.getLogger("pajisaw")
_Target = int


class _Split(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"  # NOTE: torchvision does not support the test split

    @property
    def length(self) -> float:
        split_lengths = {
            _Split.TRAIN: 0.8,  # percentage of the dataset
            _Split.VAL: 0.1,
            _Split.TEST: 0.1,
        }
        return split_lengths[self]


class PapyJigSaw(VisionDataset):
    Target = Union[_Target]
    Split = Union[_Split]

    def __init__(
        self,
        root: str,
        split: "PapyJigSaw.Split",
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        p_negative: float = 0.5,
        p_negative_in_same_img: float = 0.7,
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)
        self._split = split
        self.dataset_path = os.path.join(root, split.value)
        self._p_negative = p_negative
        self._p_negative_in_same_img = p_negative_in_same_img
        self.cache_file = os.path.join(self.dataset_path, f'{self._split}.cache')

        self.entries = {}
        self.entry_id_map = {}

    @property
    def split(self) -> "PapyJigSaw.Split":
        return self._split

    def load_entries(self):
        if not os.path.exists(self.cache_file):
            raise Exception('Entries file does not exists!')
        data = torch.load(self.cache_file)
        self.entries = data['entries']
        self.entry_id_map = data['entry_map']

    def generate_entries(self):
        if os.path.exists(self.cache_file):
            return
        fragment_paths = glob.glob(os.path.join(self.dataset_path, '**', '*.jpeg'), recursive=True)
        fragment_map = {}
        for fragment_path in fragment_paths:
            image_name = os.path.basename(os.path.dirname(fragment_path))
            fragment_name = os.path.splitext(os.path.basename(fragment_path))[0]
            col, row = map(int, fragment_name.split("_"))
            fragment_map.setdefault(image_name, []).append({'img': fragment_path, 'col': col, 'row': row,
                                                            'name': image_name,
                                                            'positive': [], 'negative': []})

        entries = {}
        for image_name, fragments in fragment_map.items():
            for first in fragments:
                for second in fragments:
                    if first['img'] == second['img']:
                        continue
                    if first['col'] == second['col'] and abs(first['row'] - second['row']) == 1:
                        first['positive'].append(second)
                    elif first['row'] == second['row'] and abs(first['col'] - second['col']) == 1:
                        first['positive'].append(second)
                    else:
                        first['negative'].append(second)
                if len(first['positive']) > 0:
                    entries.setdefault(image_name, []).append(first)
        entry_map = {i: k for i, k in enumerate(entries.keys())}
        torch.save({
            'entries': entries,
            'entry_map': entry_map
        }, self.cache_file)

    def __getitem__(self, index: int):
        if len(self.entries) == 0:
            self.load_entries()
        entry = random.choice(self.entries[self.entry_id_map[index]])
        if self._p_negative < torch.rand(1):
            target_entry = random.choice(entry['positive'])
            label = 1.
        else:
            if self._p_negative_in_same_img > torch.rand(1) and len(entry['negative']) > 0:
                target_entry = random.choice(entry['negative'])
            else:
                target_im_name = entry['name']
                while target_im_name == entry['name']:
                    target_im_name = random.choice(list(self.entries.keys()))
                target_entry = random.choice(self.entries[target_im_name])
            label = 0.

        with Image.open(entry['img']) as f:
            img = f.convert('RGB')

        with Image.open(target_entry['img']) as f:
            target_img = f.convert('RGB')

        if self.transform is not None:
            img, target_img = self.transform(img, target_img)

        assert isinstance(img, torch.Tensor)
        assert isinstance(target_img, torch.Tensor)

        stacked_img = torch.stack([img, target_img], dim=0)
        return stacked_img, torch.tensor(label, dtype=torch.float32)

    def __len__(self) -> int:
        if len(self.entries) == 0:
            self.load_entries()

        return len(self.entries)

