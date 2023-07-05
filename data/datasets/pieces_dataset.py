import logging
from typing import Callable, Optional, Union, List

import cv2
import torch
import torchvision
from torchvision.datasets import VisionDataset
import numpy as np
from paikin_tal_solver.puzzle_piece import PuzzlePiece

logger = logging.getLogger("pajisaw")
_Target = int


class PiecesDataset(VisionDataset):
    Target = Union[_Target]

    def __init__(
        self,
        pieces: List[PuzzlePiece],
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        image_size=64,
        erosion_ratio=0.07,
    ) -> None:
        super().__init__('', transforms, transform, target_transform)
        self.pieces = pieces

        self.image_size = image_size
        self.erosion_ratio = erosion_ratio
        self.entries = []
        for i, _ in enumerate(pieces):
            for j, _ in enumerate(pieces):
                if i == j:
                    continue
                self.entries.append((i, j))

    def __getitem__(self, index: int):

        gap = int(self.image_size * self.erosion_ratio / 2)
        cropper = torchvision.transforms.CenterCrop((self.image_size - gap, self.image_size - gap))
        img_converter = torchvision.transforms.Compose([
            lambda x: cv2.cvtColor(x, cv2.COLOR_LAB2RGB),
            torchvision.transforms.ToPILImage(),
            lambda x: cropper(x)
        ])

        i, j = self.entries[index]

        first_piece = self.pieces[i]
        secondary_piece = self.pieces[j]

        first_img = img_converter(first_piece.lab_image)
        second_img = img_converter(secondary_piece.lab_image)

        if self.transform is not None:
            first_img, second_img = self.transform(first_img, second_img)

        assert isinstance(first_img, torch.Tensor)
        assert isinstance(second_img, torch.Tensor)

        stacked_img = torch.stack([first_img, second_img], dim=0)
        label = index
        return stacked_img, torch.tensor(label, dtype=torch.int32)

    def __len__(self) -> int:
        return len(self.entries)

