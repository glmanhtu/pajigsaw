import json
import logging
import os
import random
from enum import Enum
from typing import Callable, Optional, Union

import cv2
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import VisionDataset

from paikin_tal_solver.puzzle_piece import PuzzlePiece

logger = logging.getLogger("pajisaw")
_Target = int


class _Split(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"

    def is_train(self):
        return self.value == 'train'

    @staticmethod
    def from_string(name):
        for key in _Split:
            if key.value == name:
                return key


class Pajigsaw(VisionDataset):
    Target = Union[_Target]
    Split = Union[_Split]

    def __init__(
        self,
        root: str,
        split: "Pajigsaw.Split",
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        image_size=512,
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)
        with open(os.path.join(root, f'{split.value}.json')) as f:
            dataset = json.load(f)
        records = {}
        for img_name in dataset:
            records[img_name] = []
            for fragment in dataset[img_name]['Fragment1v1Rotate90']:
                if fragment['degree'] == 0:
                    item = {**fragment, 'positive': [], 'negative': []}
                    records[img_name].append(item)
        self._split = split

        entries = {}
        samples = []
        for image_name, fragments in records.items():
            for first in fragments:
                for second in fragments:
                    if second['white_percentage'] > 0.85:
                        continue
                    if first['im_path'] == second['im_path']:
                        continue
                    if first['col'] == second['col'] and abs(first['row'] - second['row']) == 1:
                        first['positive'].append(second)
                    elif first['row'] == second['row'] and abs(first['col'] - second['col']) == 1:
                        first['positive'].append(second)
                    else:
                        first['negative'].append(second)
                if len(first['positive']) > 0:
                    first['im_name'] = image_name
                    entries.setdefault(image_name, []).append(first)
                    samples.append(first)
        self.im_names = sorted(entries.keys())
        self.samples = sorted(samples, key=lambda x: (x['col'], x['row']))
        self.entries = entries

    @property
    def split(self) -> "Pajigsaw.Split":
        return self._split

    def __getitem__(self, index: int):
        first_entry = self.samples[index]
        im_name = first_entry['im_name']
        if 0.75 > torch.rand(1):
            second_entry = random.choice(first_entry['positive'])
            if first_entry['col'] == second_entry['col']:
                if first_entry['row'] < second_entry['row']:
                    label = [0., 1., 0., 0.]
                else:
                    label = [0., 0., 0., 1.]
            elif first_entry['row'] == second_entry['row']:
                if first_entry['col'] < second_entry['col']:
                    label = [1., 0., 0., 0.]
                else:
                    label = [0., 0., 1., 0.]
            else:
                raise Exception(f"Incorrect entries {first_entry} and {second_entry}")

        else:
            if 0.5 > torch.rand(1) and len(first_entry['negative']) > 0:
                second_entry = random.choice(first_entry['negative'])
            else:
                target_im_name = im_name
                while target_im_name == im_name:
                    target_im_name = random.choice(self.im_names)
                second_entry = random.choice(self.entries[target_im_name])
            label = [0., 0., 0., 0.]

        first_img_path = os.path.join(self.root, first_entry['im_path'])
        with Image.open(first_img_path) as f:
            first_img = f.convert('RGB')

        second_img_path = os.path.join(self.root, second_entry['im_path'])
        with Image.open(second_img_path) as f:
            second_img = f.convert('RGB')

        if self.transform is not None:
            first_img = self.transform(first_img)
            second_img = self.transform(second_img)

        assert isinstance(first_img, torch.Tensor)
        assert isinstance(second_img, torch.Tensor)

        stacked_img = torch.stack([first_img, second_img], dim=0)
        return stacked_img, torch.tensor(label, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.samples)


class PajigsawPieces(Dataset):
    def __init__(
        self,
        root: str,
        split: "Pajigsaw.Split"
    ) -> None:
        with open(os.path.join(root, f'{split.value}.json')) as f:
            dataset = json.load(f)
        records = {}
        for img_name in dataset:
            records[img_name] = []
            for fragment in dataset[img_name]['Fragment1v1Rotate90']:
                if fragment['degree'] == 0:
                    records[img_name].append(fragment)
        self._split = split
        self.root = root
        self.entries = sorted(records.keys())
        self.entry_map = records

    @property
    def split(self) -> "Pajigsaw.Split":
        return self._split

    def __getitem__(self, index: int):
        im_name = self.entries[index]
        entry = self.entry_map[self.entries[index]]
        puzzle_id = index
        numb_rows = max([x['row'] for x in entry]) + 1
        numb_cols = max([x['col'] for x in entry]) + 1
        grid_size = (numb_rows, numb_cols)
        pieces = []
        for idx, item in enumerate(entry):
            img_path = os.path.join(self.root, item['im_path'])
            img_lab = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2LAB)
            pieces.append(PuzzlePiece(puzzle_id, (item['row'], item['col']), img_lab,
                                      piece_id=idx, puzzle_grid_size=grid_size))

        return pieces, im_name, grid_size
