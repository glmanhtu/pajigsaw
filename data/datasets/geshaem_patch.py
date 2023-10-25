import glob
import logging
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

logger = logging.getLogger("pajisaw")
_Target = int


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


def add_items_to_group(items, groups):
    reference_group = {}
    for g_id, group in enumerate(groups):
        for fragment_id in items:
            if fragment_id in group and g_id not in reference_group:
                reference_group[g_id] = group

    if len(reference_group) > 0:
        reference_ids = list(reference_group.keys())
        for fragment_id in items:
            reference_group[reference_ids[0]].add(fragment_id)
        for g_id in reference_ids[1:]:
            for fragment_id in reference_group[g_id]:
                reference_group[reference_ids[0]].add(fragment_id)
            del groups[g_id]
    else:
        groups.append(set(items))


def extract_relations(dataset_path):
    """
    There are some fragments that the papyrologists have put together by hand in the database. These fragments
    are named using the pattern of <fragment 1>_<fragment 2>_<fragment 3>...
    Since they belong to the same papyrus, we should put them to the same category
    @param dataset_path:
    """

    groups = []

    for dir_name in sorted(os.listdir(dataset_path)):
        name_components = dir_name.split("_")
        add_items_to_group(name_components, groups)

    return groups


def single_image_split(image: Image, image_size: int):
    if image.width > image.height:
        n_cols, n_rows = 2, 1
    else:
        n_cols, n_rows = 1, 2

    patches = transforms.crop(image, n_cols, n_rows)
    results = []
    cropper = torchvision.transforms.RandomCrop(image_size, pad_if_needed=True, fill=255)
    for patch in patches:
        results.append(cropper(patch))
    return tuple(results)


def two_images_split(image1: Image, image2: Image, image_size: int):
    cropper = torchvision.transforms.RandomCrop(image_size, pad_if_needed=True, fill=255)
    return cropper(image1), cropper(image2)


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
        repeat=1,
        min_size_limit=120
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)
        self._split = split
        self.root_dir = root

        self.image_size = image_size

        self.min_size_limit = min_size_limit
        self.repeat = repeat

        groups = extract_relations(root)
        self.fragment_to_group = {}
        for idx, group in enumerate(groups):
            if len(group) < 2 and not split.is_train():
                # We only evaluate the fragments that we know they are belongs to a certain groups
                # If the group have only one element, which means that very likely that we don't know
                # which group this element belongs to, so we skip it
                continue
            for fragment in group:
                self.fragment_to_group[fragment] = idx

        self.dataset = self.load_dataset()
        fragments = set([])
        for idx, items in enumerate(self.dataset):
            fragment_name = os.path.basename(os.path.dirname(items[0]))
            fragments.add(fragment_name)

        self.fragments = sorted(fragments)
        self.fragment_idx = {x: i for i, x in enumerate(self.fragments)}

    def get_group_id(self, idx):
        fragment = self.fragments[idx].split("_")[0]
        return self.fragment_to_group[fragment]

    def load_dataset(self):
        images = {}
        for img_path in glob.iglob(os.path.join(self.root_dir, '**', '*.jpg'), recursive=True):
            width, height = imagesize.get(img_path)
            if width < self.min_size_limit or height < self.min_size_limit:
                continue
            image_name = os.path.basename(os.path.dirname(img_path))
            fragment_ids = image_name.split("_")
            if len(fragment_ids) > 1 and self.split.is_train():
                continue
            if fragment_ids[0] not in self.fragment_to_group:
                continue
            images.setdefault(image_name, {})
            image_type = os.path.basename(img_path).rsplit("_", 1)[1].split('-')[0]
            images[image_name].setdefault(list(image_type)[-1], []).append(img_path)

        results = []
        for img_name in list(images.keys()):
            for img_type in list(images[img_name].keys()):
                if len(images[img_name][img_type]) > 1:
                    results.append(images[img_name][img_type])
                max_size_img = max(imagesize.get(images[img_name][img_type][0]))
                if max_size_img > self.image_size * 1.5:
                    results.append(images[img_name][img_type])
                else:
                    del images[img_name][img_type]
            if len(images[img_name].keys()) == 0:
                del images[img_name]

        return results

    @property
    def split(self) -> "GeshaemPatch.Split":
        return self._split

    def __getitem__(self, index: int):
        if index >= len(self.dataset):
            index = index % len(self.dataset)

        image_group = self.dataset[index]
        image_path = random.choice(image_group)
        fragment_name = os.path.basename(os.path.dirname(image_path))
        fragment_id = self.fragment_idx[fragment_name]
        with Image.open(image_path) as f:
            image = f.convert('RGB')

        img_transforms = torchvision.transforms.Compose([])
        if self.split.is_train():
            train_transform = A.Compose(
                [
                    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=10, p=0.5),
                ]
            )
            img_transforms = torchvision.transforms.Compose([
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.RandomVerticalFlip(),
                lambda x: np.array(x),
                lambda x: train_transform(image=x)['image'],
                torchvision.transforms.ToPILImage(),
            ])

        img_split_fn_candidates = []
        if max(image.width, image.height) > self.image_size:
            def single_split_fn():
                return single_image_split(img_transforms(image), self.image_size)
            img_split_fn_candidates.append(single_split_fn)

        if len(image_group) > 1:
            def two_split_fn():
                image2_path = image_path
                while image2_path == image_path:
                    image2_path = random.choice(image_group)
                with Image.open(image2_path) as f:
                    image2 = f.convert('RGB')
                return two_images_split(img_transforms(image), img_transforms(image2), self.image_size)
            img_split_fn_candidates.append(two_split_fn)

        split_fn = random.choice(img_split_fn_candidates)
        first_img, second_img = split_fn()

        if self.split.is_train():
            if 0.5 > torch.rand(1):
                first_img, second_img = second_img, first_img
            color_jitter = torchvision.transforms.RandomApply([
                torchvision.transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            ], p=0.5)
            first_img, second_img = color_jitter(first_img), color_jitter(second_img)

        if self.transform is not None:
            first_img, second_img = self.transform(first_img, second_img)

        assert isinstance(first_img, torch.Tensor)
        assert isinstance(second_img, torch.Tensor)

        stacked_img = torch.stack([first_img, second_img], dim=0)
        return stacked_img, fragment_id

    def __len__(self) -> int:
        return len(self.dataset) * self.repeat

