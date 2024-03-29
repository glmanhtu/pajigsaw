import glob
import logging
import math
import os
import random
from enum import Enum
from typing import Callable, Optional, Union

import albumentations as A
import imagesize
import numpy as np
import torch
import torchvision
from PIL import Image
from torchvision.datasets import VisionDataset

from data import transforms
from data.transforms import CustomRandomCrop
from data.utils import UnableToCrop

logger = logging.getLogger("pajisaw")
_Target = int


excluded = ["0567n_IRR.jpg", "0567p_IRR.jpg", "0567q_IRR.jpg", "0567t_IRR.jpg", "2881f_IRR.jpg"
            "0810a_IRR.jpg", "1306g_IRR.jpg", "1322i_IRR.jpg", "1374e_IRR.jpg", "1378d_IRR.jpg",
            "1378f_IRR.jpg", "2733t_IRR.jpg", "2849e_IRR.jpg", "2867e_IRR.jpg", "2867g_IRR.jpg",
            "2843a_IRR.jpg", "2881f_IRR.jpg", "0810a_IRR.jpg", "1290u_IRR.jpg", "2842c_IRR.jpg",
            "2849b_IRR.jpg", "2859a_IRR.jpg", "2901g_IRR.jpg", "2967c_IRR.jpg"]


class _Split(Enum):
    TRAIN = "train"
    VAL = "validation"

    def is_train(self):
        return self.value == 'train'

    @staticmethod
    def from_string(name):
        for key in _Split:
            if key.value == name:
                return key


class GeshaemPatch(VisionDataset):
    Target = Union[_Target]
    Split = Union[_Split]

    def __init__(
        self,
        root: str,
        split: "GeshaemPatch.Split",
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        image_size=512,
        erosion_ratio=0.07,
        with_negative=False,
        repeat=1,
        min_size_limit=290
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
        self.min_size_limit = min_size_limit
        self.repeat = repeat
        self.dataset = self.load_dataset()

    def load_dataset(self):
        images = []
        for img in glob.iglob(os.path.join(self.root_dir, '**', '*_IRR.jpg'), recursive=True):
            if os.path.basename(img) in excluded:
                continue
            width, height = imagesize.get(img)
            if width < self.min_size_limit or height < self.min_size_limit:
                continue
            if width < self.image_size * 2 and height < self.image_size * 2:
                continue
            images.append(img)
        images = sorted(images)
        if self.split.is_train():
            return images[:int(0.8 * len(images))]
        else:
            return images[int(0.8 * len(images)):]

    @property
    def split(self) -> "GeshaemPatch.Split":
        return self._split

    def __getitem__(self, index: int):
        if index >= len(self.dataset):
            index = index % len(self.dataset)

        with Image.open(self.dataset[index]) as f:
            image = f.convert('RGB')

        cropper_candidates = []
        height, width = image.height, image.width
        if image.width > self.image_size * 2:
            width = random.randint(self.image_size * 2, min(self.image_size * 4, image.width))
            if height > self.image_size:
                height = random.randint(self.image_size, min(width // 2, image.height))
            cropper = CustomRandomCrop((height, width))
            cropper_candidates.append(cropper)

        height, width = image.height, image.width
        if image.height > self.image_size * 2:
            height = random.randint(self.image_size * 2, min(self.image_size * 4, image.height))
            if width > self.image_size:
                width = random.randint(self.image_size, min(height // 2, image.width))
            cropper = CustomRandomCrop((height, width))
            cropper_candidates.append(cropper)

        cropper = random.choice(cropper_candidates)
        image = cropper(image)
        label = [1, 0, 0, 0]
        if image.height > image.width:
            label = [0, 1, 0, 0]

        if self.split.is_train():
            train_transform = A.Compose(
                [
                    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.15, rotate_limit=20, p=0.5),
                    A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
                ]
            )
            img_transforms = torchvision.transforms.Compose([
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.RandomVerticalFlip(),
                lambda x: np.array(x),
                lambda x: train_transform(image=x)['image'],
                torchvision.transforms.ToPILImage(),
            ])

            image = img_transforms(image)

        patches = transforms.split_with_gap(image, 0.5, 0)
        min_size = min([min(x.width, x.height) for x in patches])
        first_img, second_img = patches[0], patches[1]
        erosion_ratio = self.erosion_ratio
        if self._split.is_train():
            erosion_ratio = random.uniform(self.erosion_ratio, self.erosion_ratio * 3)
        piece_size_erosion = math.ceil(min_size * (1 - erosion_ratio))
        cropper = torchvision.transforms.RandomCrop(piece_size_erosion)
        first_img, second_img = cropper(first_img), cropper(second_img)

        if 0.5 > torch.rand(1):
            first_img, second_img = second_img, first_img
            if label[0] == 1:
                label = [0, 0, 1, 0]
            else:
                label = [0, 0, 0, 1]

        if 0.3 > torch.rand(1):
            while True:
                try:
                    other_index = random.choice([i for i in range(len(self.dataset)) if i != index])
                    with Image.open(self.dataset[other_index]) as f:
                        other_image = f.convert('RGB')
                    cropper = CustomRandomCrop((second_img.height, second_img.width))
                    second_img = cropper(other_image)
                    label = [0, 0, 0, 0]
                    break
                except UnableToCrop:
                    pass

        if self.transform is not None:
            first_img, second_img = self.transform(first_img, second_img)

        assert isinstance(first_img, torch.Tensor)
        assert isinstance(second_img, torch.Tensor)

        stacked_img = torch.stack([first_img, second_img], dim=0)
        return stacked_img, torch.tensor(label, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.dataset) * self.repeat

