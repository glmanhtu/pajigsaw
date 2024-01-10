import argparse
import logging

import albumentations as A
import cv2
import numpy as np
import torchvision.transforms

from data.datasets.michigan_dataset import MichiganDataset
from data.transforms import RandomSizedCrop, ACompose

logging.basicConfig(level=logging.INFO, format='%(asctime)s :: %(levelname)s :: %(message)s')

parser = argparse.ArgumentParser('Pajigsaw testing script', add_help=False)
parser.add_argument('--data-path', required=True, type=str, help='path to dataset')
args = parser.parse_args()


patch_size = 512
transform = torchvision.transforms.Compose([
    # ACompose([
    #     A.ShiftScaleRotate(shift_limit=0., scale_limit=0.1, rotate_limit=10, p=0.5, value=(255, 255, 255),
    #                        border_mode=cv2.BORDER_CONSTANT),
    # ]),
    torchvision.transforms.RandomApply([
        RandomSizedCrop(min_width=224, min_height=224, pad_if_needed=True, fill=(255, 255, 255)),
    ]),
    torchvision.transforms.RandomCrop(512, pad_if_needed=True, fill=(255, 255, 255)),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.RandomVerticalFlip(),
    ACompose([
        A.CoarseDropout(max_holes=16, min_holes=1, min_height=16, max_height=128, min_width=16, max_width=128,
                        fill_value=255, always_apply=True),
    ]),
    torchvision.transforms.Resize(patch_size),
    torchvision.transforms.RandomApply([
        torchvision.transforms.ColorJitter(brightness=0.2, contrast=0.3, saturation=0.3, hue=0.1),
    ], p=.5),
    torchvision.transforms.RandomApply([
        torchvision.transforms.GaussianBlur((3, 3), (1.0, 2.0)),
    ], p=.5),
    torchvision.transforms.RandomGrayscale(p=0.2),
])

train_dataset = MichiganDataset(args.data_path, MichiganDataset.Split.TRAIN, transforms=transform)
un_normaliser = torchvision.transforms.Compose([
    lambda x: np.asarray(x)
])
for img, label in train_dataset:
    first_img = un_normaliser(img)
    # if label[0] == 1:
    #     image = np.concatenate([first_img, second_img], axis=1)
    # elif label[2] == 1:
    #     image = np.concatenate([second_img, first_img], axis=1)
    # elif label[3] == 1:
    #     image = np.concatenate([second_img, first_img], axis=0)
    # elif label[1] == 1:
    #     image = np.concatenate([first_img, second_img], axis=0)
    #
    # else:
    #     image = np.concatenate([first_img, np.zeros_like(first_img), second_img], axis=0)

    # image = cv2.bitwise_not(image)
    cv2.imshow('image', cv2.cvtColor(first_img, cv2.COLOR_RGB2BGR))

    # waitKey() waits for a key press to close the window and 0 specifies indefinite loop
    cv2.waitKey(500)

    # cv2.destroyAllWindows() simply destroys all the windows we created.
cv2.destroyAllWindows()
