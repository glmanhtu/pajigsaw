import logging
from typing import Callable, Optional, Union, List

import cv2
import torch
import torchvision
from torchvision.datasets import VisionDataset
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
    ) -> None:
        super().__init__('', transforms, transform, target_transform)
        self.pieces = pieces

        self.entries = []
        for i, _ in enumerate(pieces):
            for j, _ in enumerate(pieces):
                if i == j:
                    continue
                self.entries.append((i, j))

    def __getitem__(self, index: int):
        img_converter = torchvision.transforms.Compose([
            lambda x: cv2.cvtColor(x, cv2.COLOR_LAB2RGB),
            torchvision.transforms.ToPILImage(),
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


class PiecesDatasetTriplet(VisionDataset):
    Target = Union[_Target]

    def __init__(
        self,
        pieces: List[PuzzlePiece],
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__('', transforms, transform, target_transform)
        self.pieces = pieces

        self.entries = []
        for i, _ in enumerate(pieces):
            for j, _ in enumerate(pieces):
                if i == j:
                    continue
                self.entries.append((i, j))

    def __getitem__(self, index: int):
        img_converter = torchvision.transforms.Compose([
            lambda x: cv2.cvtColor(x, cv2.COLOR_LAB2RGB),
            torchvision.transforms.ToPILImage(),
        ])

        i, j = self.entries[index]

        first_piece = self.pieces[i]
        secondary_piece = self.pieces[j]

        first_img = img_converter(first_piece.lab_image)
        second_img = img_converter(secondary_piece.lab_image)

        images = []

        # Matching on the right of the first img
        first_tensor, second_tensor = self.transform(first_img, second_img.rotate(180))
        images.append(torch.stack([first_tensor, second_tensor], dim=0))

        # Matching on the bottom of the first img
        first_tensor, second_tensor = self.transform(first_img.rotate(90), second_img.rotate(270))
        images.append(torch.stack([first_tensor, second_tensor], dim=0))

        # Matching on the left of the first img
        first_tensor, second_tensor = self.transform(first_img.rotate(180), second_img)
        images.append(torch.stack([first_tensor, second_tensor], dim=0))

        # Matching on the top of the first img
        first_tensor, second_tensor = self.transform(first_img.rotate(270), second_img.rotate(90))
        images.append(torch.stack([first_tensor, second_tensor], dim=0))

        stacked_img = torch.cat(images, dim=0)
        label = index

        return stacked_img, torch.tensor(label, dtype=torch.int32)

    def __len__(self) -> int:
        return len(self.entries)

