import glob
import os
from enum import Enum
from typing import Callable, Optional, Union

import imagesize
import torch
from PIL import Image
from ml_engine.data.grouping import add_items_to_group
from torchvision.datasets import VisionDataset

_Target = int


class _Split(Enum):
    TRAIN = "train"
    VAL = "validation"

    def is_train(self):
        return self.value == 'train'

    def is_val(self):
        return self.value == 'validation'

    @staticmethod
    def from_string(name):
        for key in _Split:
            if key.value == name:
                return key


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
        include_verso=False,
        min_size_limit=300
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)
        self._split = split
        self.root_dir = root

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

        self.dataset, fragments = self.load_dataset(include_verso)

        self.fragments = sorted(fragments)
        self.fragment_idx = {x: i for i, x in enumerate(self.fragments)}

        indicates = torch.arange(len(fragments)).type(torch.int)
        pairs = torch.combinations(indicates, r=2, with_replacement=True)
        self.pairs = pairs

    def get_fragment_idx(self, image_name: str) -> int:
        fragment_id = image_name.split("_")[0]
        return self.fragment_idx[fragment_id]

    def load_dataset(self, include_verso):
        images = []
        fragments = set([])
        for img_path in sorted(glob.glob(os.path.join(self.root_dir, '**', '*.jpg'), recursive=True)):
            image_name = os.path.basename(os.path.dirname(os.path.dirname(img_path)))
            fragment_ids = image_name.split("_")
            if len(fragment_ids) > 1:
                # We exclude the assembled fragments to prevent data leaking
                continue
            if fragment_ids[0] not in self.fragment_to_group:
                continue

            image_type = os.path.basename(os.path.dirname(img_path)).rsplit("_", 1)[1].split('-')[0]
            image_type = list(image_type)[-1]
            if image_type.upper() == 'V' and not include_verso:
                continue

            images.append(img_path)
            fragments.add(image_name)

        return images, fragments

    @property
    def split(self) -> "GeshaemPatch.Split":
        return self._split

    def __getitem__(self, index: int):
        x1_id, x2_id = tuple(self.pairs[index])
        img_path = self.dataset[x1_id.item()]

        with Image.open(img_path) as f:
            image = f.convert('RGB')

        img2_path = self.dataset[x2_id.item()]
        with Image.open(img2_path) as f:
            image2 = f.convert('RGB')

        if self.transform:
            image = self.transform(image)
            image2 = self.transform(image2)
        stacked_img = torch.stack([image, image2], dim=0)
        return stacked_img, self.pairs[index]

    def __len__(self) -> int:
        return len(self.pairs)

