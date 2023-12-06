import glob
import logging
import os
from enum import Enum
from typing import Callable, Optional, Union

import imagesize
import torch
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
        fragment_type='R',  # Recto or Verso
        min_size_limit=300
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
                for fragment2 in group:
                    self.fragment_to_group.setdefault(fragment, set([])).add(fragment2)

        self.dataset, fragments = self.load_dataset(fragment_type)

        self.fragments = sorted(fragments)
        self.fragment_idx = {x: i for i, x in enumerate(self.fragments)}
        self.data_labels = []
        for item in self.dataset:
            fragment_name = os.path.basename(os.path.dirname(item))
            self.data_labels.append(self.fragment_idx[fragment_name])

    def load_dataset(self, fragment_type):
        images = []
        fragments = set([])
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

            # image_type = os.path.basename(img_path).rsplit("_", 1)[1].split('-')[0]
            # image_type = list(image_type)[-1]

            images.append(img_path)
            fragment_name = os.path.basename(os.path.dirname(img_path))
            fragments.add(fragment_name)

        return images, fragments

    @property
    def split(self) -> "GeshaemPatch.Split":
        return self._split

    def __getitem__(self, index: int):
        img_path = self.dataset[index]
        fragment_name = os.path.basename(os.path.dirname(img_path))
        fragment_id = self.fragment_idx[fragment_name]

        with Image.open(img_path) as f:
            image = f.convert('RGB')

        if self.transform is not None:
            image = self.transform(image)

        assert isinstance(image, torch.Tensor)

        return image, fragment_id

    def __len__(self) -> int:
        return len(self.dataset)

