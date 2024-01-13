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

from data import transforms

logger = logging.getLogger("pajisaw")
_Target = int


class _Split(Enum):
    TRAIN = "train"
    VAL = "validation"

    @property
    def sub_dir(self) -> str:
        paths = {
            _Split.TRAIN: 'DIV2K_train_HR',  # percentage of the dataset
            _Split.VAL: 'DIV2K_valid_HR',
        }
        return paths[self]

    def is_train(self):
        return self.value == 'train'

    @staticmethod
    def from_string(name):
        for key in _Split:
            if key.value == name:
                return key


class DIV2KPatch(VisionDataset):
    Target = Union[_Target]
    Split = Union[_Split]

    def __init__(
        self,
        root: str,
        split: "DIV2KPatch.Split",
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        image_size=64,
        erosion_ratio=0.07,
        with_negative=False,
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)
        self._split = split
        self.root_dir = root

        self.image_size = image_size
        self.with_negative = with_negative
        self.erosion_ratio = erosion_ratio

        self.cropper_class = torchvision.transforms.RandomCrop
        if not split.is_train():
            self.cropper_class = torchvision.transforms.CenterCrop
        self.dataset = self.load_dataset()

    @property
    def split(self) -> "DIV2KPatch.Split":
        return self._split

    def load_dataset(self):
        dataset_dir = os.path.join(self.root_dir, self.split.sub_dir)
        images = []
        for root, dirs, files in os.walk(dataset_dir):
            for file in files:
                if file.lower().endswith((".jpg", ".png")):
                    images.append(os.path.join(root, file))
        return images

    def read_image(self, index):
        img_path = self.dataset[index]
        with Image.open(img_path) as f:
            image = f.convert('RGB')

        if self.split.is_train():
            train_transform = A.Compose(
                [
                    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.15, rotate_limit=20, p=0.5),
                    A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
                ]
            )
            transforms = torchvision.transforms.Compose([
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.RandomVerticalFlip(),
                lambda x: np.array(x),
                lambda x: train_transform(image=x)['image'],
                torchvision.transforms.ToPILImage(),
            ])

            image = transforms(image)

        return image

    def __getitem__(self, index: int):
        image = self.read_image(index)
        cropper = self.cropper_class((self.image_size * 2, self.image_size * 3))
        patch = cropper(image)

        # Crop the image into a grid of 3 x 2 patches
        crops = transforms.crop(patch, 3, 2)
        erosion_ratio = self.erosion_ratio
        if self._split.is_train():
            erosion_ratio = random.uniform(self.erosion_ratio, self.erosion_ratio * 2)
        piece_size_erosion = math.ceil(self.image_size * (1 - erosion_ratio))
        cropper = torchvision.transforms.CenterCrop(piece_size_erosion)
        first_img = cropper(crops[0])

        # Second image is next to the first image
        second_img = cropper(crops[1])

        # Third image is right below the second image
        third_img = cropper(crops[4])

        # Fourth mage is right below the first image
        fourth_img = cropper(crops[3])

        label = [1., 0., 0., 0.]
        if self.with_negative and 0.3 > torch.rand(1):
            if 0.5 < torch.rand(1):
                second_img, third_img = third_img, second_img
            else:
                second_img = cropper(crops[2])

            if 0.5 < torch.rand(1):
                second_img, first_img = first_img, second_img

            label = [0., 0., 0., 0.]

        else:
            if 0.5 < torch.rand(1):
                second_img, fourth_img = fourth_img, second_img
                label = [0., 1., 0., 0.]

            if 0.5 < torch.rand(1):
                first_img, second_img = second_img, first_img
                if label[0] == 1:
                    label = [0., 0., 1., 0.]
                else:
                    label = [0., 0., 0., 1.]

        if self.transform is not None:
            first_img, second_img = self.transform(first_img, second_img)

        assert isinstance(first_img, torch.Tensor)
        assert isinstance(second_img, torch.Tensor)

        stacked_img = torch.stack([first_img, second_img], dim=0)
        return stacked_img, torch.tensor(label, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.dataset)


class Div2kPatchTriplet(DIV2KPatch):
    def __getitem__(self, index: int):
        image = self.read_image(index)
        cropper = self.cropper_class((self.image_size * 2, self.image_size * 3))
        patch = cropper(image)

        # Crop the image into a grid of 3 x 2 patches
        crops = transforms.crop(patch, 3, 2)
        erosion_ratio = self.erosion_ratio
        if self._split.is_train():
            erosion_ratio = random.uniform(self.erosion_ratio, self.erosion_ratio * 2)
        piece_size_erosion = math.ceil(self.image_size * (1 - erosion_ratio))
        cropper = torchvision.transforms.CenterCrop(piece_size_erosion)

        results = []

        # Matching on the right of the first img
        anchor = self.transform(cropper(crops[0]))
        positive = self.transform(cropper(crops[1]).rotate(180))
        negative = self.transform(cropper(crops[1]))

        results.append(torch.stack([anchor, positive, negative], dim=0))

        # Matching on the left of the first img
        anchor = self.transform(cropper(crops[5]).rotate(180))
        positive = self.transform(cropper(crops[4]))
        negative = self.transform(cropper(crops[1]))

        results.append(torch.stack([anchor, positive, negative], dim=0))

        # Matching on the bottom of the first img
        anchor = self.transform(cropper(crops[1]).rotate(90))
        positive = self.transform(cropper(crops[4]).rotate(270))
        negative = self.transform(cropper(crops[3]))

        results.append(torch.stack([anchor, positive, negative], dim=0))

        # Matching on the top of the first img
        anchor = self.transform(cropper(crops[3]).rotate(270))
        positive = self.transform(cropper(crops[1]).rotate(90))
        negative = self.transform(cropper(crops[2]))

        results.append(torch.stack([anchor, positive, negative], dim=0))
        return torch.stack(results), index


