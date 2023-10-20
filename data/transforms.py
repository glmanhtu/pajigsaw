import numpy as np
import torchvision
from PIL import Image
from torchvision import transforms

from misc.utils import UnableToCrop


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


class CustomRandomCrop:
    def __init__(self, crop_size, white_percentage_limit=0.6, max_retry=1000, im_path=''):
        self.cropper = torchvision.transforms.RandomCrop(crop_size, pad_if_needed=True, fill=255)
        self.white_percentage_limit = white_percentage_limit
        self.max_retry = max_retry
        self.im_path = im_path

    def crop(self, img):
        current_retry = 0
        while current_retry < self.max_retry:
            out = self.cropper(img)
            if compute_white_percentage(out) <= self.white_percentage_limit:
                return out
            current_retry += 1
        raise UnableToCrop('Unable to crop ', im_path=self.im_path)

    def __call__(self, img):
        return self.crop(img)
