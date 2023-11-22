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
    TEST = "test"

    def is_train(self):
        return self.value == 'train'

    def is_test(self):
        return self.value == 'test'

    def is_val(self):
        return self.value == 'validation'

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


def single_image_split(image: Image, cropper):
    if image.width > image.height:
        n_cols, n_rows = 2, 1
    else:
        n_cols, n_rows = 1, 2

    patches = transforms.crop(image, n_cols, n_rows)
    results = []
    for patch in patches:
        results.append(cropper(patch))
    return tuple(results)


def two_images_split(image1: Image, image2: Image, cropper):
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
        min_size_limit=224
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)
        self._split = split
        self.root_dir = root

        self.image_size = image_size

        self.min_size_limit = min_size_limit

        groups = extract_relations(root)
        self.fragment_to_group = {}
        for idx, group in enumerate(groups):
            if len(group) < 2 and split.is_val():
                # We only evaluate the fragments that we know they are belongs to a certain groups
                # If the group have only one element, which means that very likely that we don't know
                # which group this element belongs to, so we skip it
                continue
            for fragment in group:
                self.fragment_to_group[fragment] = idx

        self.dataset = self.load_dataset()
        fragments = set([])
        for items in self.dataset:
            fragment_name = os.path.basename(os.path.dirname(items[0]))
            fragments.add(fragment_name)

        self.fragments = sorted(fragments)
        self.fragment_idx = {x: i for i, x in enumerate(self.fragments)}

    def get_group_id(self, idx):
        fragment = self.fragments[idx].split("_")[0]
        return self.fragment_to_group[fragment]

    def load_dataset(self):
        images = {}
        for img_path in sorted(glob.glob(os.path.join(self.root_dir, '**', '*.jpg'), recursive=True)):
            width, height = imagesize.get(img_path)
            if width < self.min_size_limit or height < self.min_size_limit:
                continue
            image_name = os.path.basename(os.path.dirname(img_path))
            fragment_ids = image_name.split("_")
            if len(fragment_ids) > 1 and self.split.is_train():
                # We exclude the assembled fragments in training to prevent data leaking
                continue
            if fragment_ids[0] not in self.fragment_to_group:
                continue
            images.setdefault(image_name, {})
            image_type = os.path.basename(img_path).rsplit("_", 1)[1].split('-')[0]
            # image_type is including Recto (R) and Verso (V)
            images[image_name].setdefault(list(image_type)[-1], []).append(img_path)

        results = []
        for img_name in list(images.keys()):
            for img_type in list(images[img_name].keys()):
                max_size_img = max(imagesize.get(images[img_name][img_type][0]))
                if len(images[img_name][img_type]) > 1:
                    # If there is more than one type of image (COLV | COLR | IRR | IRV)
                    results.append(images[img_name][img_type])
                elif max_size_img > self.image_size * 1.5:
                    # If we can split the image into at least 2 patches
                    results.append(images[img_name][img_type])
                else:
                    # Otherwise, exclude the image
                    del images[img_name][img_type]
            if len(images[img_name].keys()) == 0:
                del images[img_name]

        return results

    @property
    def split(self) -> "GeshaemPatch.Split":
        return self._split

    def __getitem__(self, index: int):
        image_group = self.dataset[index]
        first_img_path = random.choice(image_group)
        fragment_name = os.path.basename(os.path.dirname(first_img_path))
        fragment_id = self.fragment_idx[fragment_name]
        with Image.open(first_img_path) as f:
            first_img = f.convert('RGB')

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
                torchvision.transforms.RandomApply([
                    torchvision.transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                ], p=0.5)
            ])

        img_split_fn_candidates = []
        cropper = torchvision.transforms.RandomCrop(self.image_size, pad_if_needed=True, fill=255)
        if max(first_img.width, first_img.height) > self.image_size:
            def single_split_fn():
                return single_image_split(img_transforms(first_img), cropper)
            img_split_fn_candidates.append(single_split_fn)

        if len(image_group) > 1:
            def two_split_fn():
                second_img_path = first_img_path
                while second_img_path == first_img_path:
                    second_img_path = random.choice(image_group)
                with Image.open(second_img_path) as f:
                    second_img = f.convert('RGB')
                return two_images_split(img_transforms(first_img), img_transforms(second_img), cropper)
            img_split_fn_candidates.append(two_split_fn)

        split_fn = random.choice(img_split_fn_candidates)
        first_img, second_img = split_fn()

        if self.transform is not None:
            first_img, second_img = self.transform(first_img, second_img)

        assert isinstance(first_img, torch.Tensor)
        assert isinstance(second_img, torch.Tensor)

        stacked_img = torch.stack([first_img, second_img], dim=0)
        return stacked_img, fragment_id

    def __len__(self) -> int:
        return len(self.dataset)

