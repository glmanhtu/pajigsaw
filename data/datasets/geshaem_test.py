import glob
import logging
import os
import random
from typing import Callable, Optional, Union

import imagesize
import torch
from PIL import Image
from torchvision.datasets import VisionDataset

from data.transforms import CustomRandomCrop

logger = logging.getLogger("pajisaw")
_Target = int


excluded = ["0567n_IRR.jpg", "0567p_IRR.jpg", "0567q_IRR.jpg", "0567t_IRR.jpg", "2881f_IRR.jpg"
            "0810a_IRR.jpg", "1306g_IRR.jpg", "1322i_IRR.jpg", "1374e_IRR.jpg", "1378d_IRR.jpg",
            "1378f_IRR.jpg", "2733t_IRR.jpg", "2849e_IRR.jpg", "2867e_IRR.jpg", "2867g_IRR.jpg",
            "2843a_IRR.jpg", "2881f_IRR.jpg", "0810a_IRR.jpg", "1290u_IRR.jpg", "2842c_IRR.jpg",
            "2849b_IRR.jpg", "2859a_IRR.jpg", "2901g_IRR.jpg", "2967c_IRR.jpg"]


class GeshaemTest(VisionDataset):
    Target = Union[_Target]

    def __init__(
        self,
        root: str,
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
        self.root_dir = root

        self.image_size = image_size
        self.with_negative = with_negative
        self.erosion_ratio = erosion_ratio

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
            if width < self.image_size and height < self.image_size:
                continue
            images.append(img)
        images = sorted(images)
        pairs = []
        for im1 in images:
            for im2 in images:
                pairs.append((im1, im2))
        return pairs

    def read_patch(self, img_path):
        with Image.open(img_path) as f:
            image = f.convert('RGB')

        height, width = image.height, image.width
        if height > self.image_size:
            height = random.randint(self.image_size, min(self.image_size * 2, height))
        if width > self.image_size:
            width = random.randint(self.image_size, min(self.image_size * 2, width))
        size = min(height, width)
        cropper = CustomRandomCrop((size, size), im_path=img_path)
        return cropper(image)

    def __getitem__(self, index: int):
        if index >= len(self.dataset):
            index = index % len(self.dataset)

        im1, im2 = self.dataset[index]

        first_img, second_img = self.read_patch(im1), self.read_patch(im2)

        if self.transform is not None:
            first_img, second_img = self.transform(first_img, second_img)

        assert isinstance(first_img, torch.Tensor)
        assert isinstance(second_img, torch.Tensor)

        stacked_img = torch.stack([first_img, second_img], dim=0)
        return stacked_img, torch.tensor(index, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.dataset) * self.repeat

