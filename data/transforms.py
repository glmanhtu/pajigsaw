import random

import numpy as np
import torchvision
from PIL import Image, ImageFilter
from torchvision import transforms
import torchvision.transforms.functional as F
import albumentations as A
from misc.utils import UnableToCrop


class TwoImgSyncEval:
    def __init__(self, image_size):
        self.normalize = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        self.image_size = image_size

    def __call__(self, first_img, second_img):

        first_img = self.normalize(first_img)
        second_img = self.normalize(second_img)

        return first_img, second_img


class ACompose:
    def __init__(self, a_transforms):
        self.transform = A.Compose(a_transforms)

    def __call__(self, image):
        np_img = np.asarray(image)
        np_img = self.transform(image=np_img)['image']
        return Image.fromarray(np_img)


class PadCenterCrop(object):
    def __init__(self, size, pad_if_needed=False, fill=0, padding_mode='constant'):
        if isinstance(size, (int, float)):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.pad_if_needed = pad_if_needed
        self.padding_mode = padding_mode
        self.fill = fill

    def __call__(self, img):

        # pad the width if needed
        if self.pad_if_needed and img.size[0] < self.size[1]:
            img = F.pad(img, (self.size[1] - img.size[0], 0), self.fill, self.padding_mode)
        # pad the height if needed
        if self.pad_if_needed and img.size[1] < self.size[0]:
            img = F.pad(img, (0, self.size[0] - img.size[1]), self.fill, self.padding_mode)

        return F.center_crop(img, self.size)


class RandomSizedCrop:

    def __init__(self, min_width, min_height, pad_if_needed=False, fill=0, padding_mode='constant'):
        self.min_width = min_width
        self.min_height = min_height
        self.pad_if_needed = pad_if_needed
        self.fill = fill
        self.padding_mode = padding_mode

    def __call__(self, image):
        width, height = image.size
        if self.min_width < image.width:
            width = random.randint(self.min_width, image.width)
        if self.min_height < image.height:
            height = random.randint(self.min_height, image.height)

        cropper = transforms.RandomCrop((height, width), pad_if_needed=self.pad_if_needed, fill=self.fill,
                                        padding_mode=self.padding_mode)
        return cropper(image)


class GaussianBlur(object):
    """
    Apply Gaussian Blur to the PIL image.
    """
    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        do_it = random.random() <= self.prob
        if not do_it:
            return img

        return img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)
            )
        )


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


def make_square(im, fill_color=(0, 0, 0)):
    x, y = im.size
    size = max(x, y)
    new_im = Image.new('RGB', (size, size), fill_color)
    new_im.paste(im, (int((size - x) / 2), int((size - y) / 2)))
    return new_im


def compute_white_percentage(img, ref_size=224):
    gray = img.convert('L')
    if gray.width > ref_size:
        gray = gray.resize((ref_size, ref_size))
    gray = np.asarray(gray)
    white_pixel_count = np.sum(gray > 250)
    total_pixels = gray.shape[0] * gray.shape[1]
    return white_pixel_count / total_pixels


class RandomResize:
    def __init__(self, img_size, ratio=(0.6, 1.0)):
        self.ratio = ratio
        self.img_size = img_size

    def __call__(self, img):
        ratio = random.uniform(self.ratio[0], self.ratio[1])
        w, h = int(img.width * ratio), int(img.height * ratio)
        cropper = torchvision.transforms.Resize((h, w))
        return cropper(img)


class CustomRandomCrop:
    def __init__(self, crop_size, white_percentage_limit=0.6, max_retry=1000, im_path=''):
        self.cropper = torchvision.transforms.RandomCrop(crop_size, pad_if_needed=True, fill=255)
        self.white_percentage_limit = white_percentage_limit
        self.max_retry = max_retry
        self.im_path = im_path

    def crop(self, img):
        current_retry = 0
        curr_w_p = 0
        while current_retry < self.max_retry:
            out = self.cropper(img)
            curr_w_p = compute_white_percentage(out)
            if curr_w_p <= self.white_percentage_limit:
                return out
            current_retry += 1
        raise UnableToCrop(f'Unable to crop, curr wp: {curr_w_p}', im_path=self.im_path)

    def __call__(self, img):
        return self.crop(img)
