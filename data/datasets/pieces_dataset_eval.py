import logging
import os
from typing import Callable, Optional, Union, List

import cv2
import torch
import torchvision
from torchvision.datasets import VisionDataset
import numpy as np

from paikin_tal_solver.puzzle_importer import Puzzle
from paikin_tal_solver.puzzle_piece import PuzzlePiece

logger = logging.getLogger("pajisaw")
_Target = int


class PuzzleDataset(VisionDataset):
    Target = Union[_Target]

    def __init__(
        self,
        root: str,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        image_size=64,
        erosion_ratio=0.07,
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)

        puzzles = []
        for root, dirs, files in os.walk(root):
            for file in files:
                if file.lower().endswith((".jpg", ".png")):
                    img_path = os.path.join(root, file)
                    puzzle = Puzzle(0, img_path, image_size, starting_piece_id=0)
                    puzzles.append(puzzle)

        self.puzzles = puzzles

        self.image_size = image_size
        self.erosion_ratio = erosion_ratio
        self.entries = []
        for p in puzzles:
            for i, _ in enumerate(p.pieces):
                for j, _ in enumerate(p.pieces):
                    if i == j:
                        continue
                    self.entries.append((p, i, j))

    def __getitem__(self, index: int):

        gap = int(self.image_size * self.erosion_ratio)
        cropper = torchvision.transforms.CenterCrop((self.image_size - gap, self.image_size - gap))
        img_converter = torchvision.transforms.Compose([
            lambda x: cv2.cvtColor(x, cv2.COLOR_LAB2RGB),
            torchvision.transforms.ToPILImage(),
            lambda x: cropper(x)
        ])

        p, i, j = self.entries[index]

        first_piece = p.pieces[i]
        secondary_piece = p.pieces[j]

        first_img = img_converter(first_piece.lab_image)
        second_img = img_converter(secondary_piece.lab_image)

        if self.transform is not None:
            first_img, second_img = self.transform(first_img, second_img)

        assert isinstance(first_img, torch.Tensor)
        assert isinstance(second_img, torch.Tensor)

        stacked_img = torch.stack([first_img, second_img], dim=0)

        first_loc, second_loc = first_piece._orig_loc, secondary_piece._orig_loc

        if first_loc[0] == second_loc[0] and second_loc[1] - first_loc[1] == 1:
            label = [1., 0., 0., 0.]
        elif first_loc[1] == second_loc[1] and second_loc[0] - first_loc[0] == 1:
            label = [0., 1., 0., 0.]
        elif first_loc[0] == second_loc[0] and first_loc[1] - second_loc[1] == 1:
            label = [0., 0., 1., 0.]
        elif first_loc[1] == second_loc[1] and first_loc[0] - second_loc[0] == 1:
            label = [0., 0., 0., 1.]
        else:
            label = [0., 0., 0., 0.]

        return stacked_img, torch.tensor(label, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.entries)

