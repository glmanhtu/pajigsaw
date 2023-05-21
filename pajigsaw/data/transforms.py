import torch
from PIL import Image, ImageOps
from torchvision import transforms

from pajigsaw.utils import utils


class TwoImgSyncAugmentation:
    def __init__(self, image_size):
        self.img_size = image_size
        self.default_transforms = transforms.Compose([
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            # utils.GaussianBlur(0.1)
        ])

        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        self.image_size = image_size

    def __call__(self, first_img, second_img):
        max_size = max(first_img.width, first_img.height, second_img.width, second_img.height)
        image_transformer = transforms.Compose([
            transforms.RandomCrop(max_size, pad_if_needed=True, fill=255),
            transforms.Resize(self.image_size),
            self.default_transforms,
        ])

        first_img = image_transformer(first_img)
        second_img = image_transformer(second_img)

        # Horizontally flipping
        if 0.5 < torch.rand(1):
            first_img = first_img.transpose(Image.FLIP_LEFT_RIGHT)
            second_img = second_img.transpose(Image.FLIP_LEFT_RIGHT)

        # Vertically flipping
        if 0.5 < torch.rand(1):
            first_img = first_img.transpose(Image.FLIP_TOP_BOTTOM)
            second_img = second_img.transpose(Image.FLIP_TOP_BOTTOM)

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
        max_size = max(first_img.width, first_img.height, second_img.width, second_img.height)
        image_transformer = transforms.Compose([
            lambda x: ImageOps.invert(x),
            transforms.CenterCrop(max_size),    # CenterCrop will pad the image, but with value 0
            lambda x: ImageOps.invert(x),
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
