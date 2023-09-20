import random

import numpy as np
from PIL import ImageOps, Image
from torchvision import transforms


class TwoImgSyncAugmentation:
    def __init__(self, image_size):
        self.img_size = image_size
        self.default_transforms = transforms.Compose([
            # transforms.RandomApply(
            #     [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2)],
            #     p=0.8
            # ),
            # transforms.RandomGrayscale(p=0.2),
            # utils.GaussianBlur(0.1)
        ])

        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        self.image_size = image_size

    def __call__(self, first_img, second_img):
        max_size = max(first_img.width, first_img.height, second_img.width, second_img.height)
        max_size = round(random.uniform(1.05, 1.15) * max_size)
        image_transformer = transforms.Compose([
            transforms.RandomCrop(max_size, pad_if_needed=True, fill=255),
            transforms.Resize(self.image_size),
            self.default_transforms,
        ])

        first_img = image_transformer(first_img)
        second_img = image_transformer(second_img)

        first_img = self.normalize(first_img)
        second_img = self.normalize(second_img)

        return first_img, second_img


class TwoImgSyncEval:
    def __init__(self, image_size):
        self.img_size = image_size

        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        self.image_size = image_size

    def __call__(self, first_img, second_img):
        image_transformer = transforms.Compose([
            transforms.Resize(self.image_size),
        ])

        first_img = image_transformer(first_img)
        second_img = image_transformer(second_img)

        first_img = self.normalize(first_img)
        second_img = self.normalize(second_img)

        return first_img, second_img


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor


def crop(im: Image, n_cols, n_rows):
    width = im.width // n_cols
    height = im.height // n_rows
    patches = []
    for i in range(n_rows):
        for j in range(n_cols):
            box = (j*width, i*height, (j+1)*width, (i+1)*height)
            patches.append(im.crop(box))
    return patches


def split_with_gap(im: Image, long_direction_ratio, gap: float):
    patches = []
    if im.width > im.height:
        box = 0, 0, int(long_direction_ratio * im.width), im.height
        patches.append(im.crop(box))
        box = int((long_direction_ratio + gap) * im.width), 0, im.width, im.height
        patches.append(im.crop(box))
    else:
        box = 0, 0, im.width, int(long_direction_ratio * im.height)
        patches.append(im.crop(box))
        box = 0, int((long_direction_ratio + gap) * im.height), im.width, im.height
        patches.append(im.crop(box))
    return patches
