import argparse
import logging

import cv2
import numpy as np
import torch
import torchvision.transforms

from data.datasets.div2k_patch import DIV2KPatch
from data.datasets.geshaem_patch import GeshaemPatch
from data.transforms import TwoImgSyncEval, UnNormalize

logging.basicConfig(level=logging.INFO, format='%(asctime)s :: %(levelname)s :: %(message)s')

parser = argparse.ArgumentParser('Pajigsaw testing script', add_help=False)
parser.add_argument('--data-path', required=True, type=str, help='path to dataset')
args = parser.parse_args()

transform = TwoImgSyncEval(224)
train_dataset = GeshaemPatch(args.data_path, split=GeshaemPatch.Split.VAL, transform=transform, with_negative=True)
un_normaliser = torchvision.transforms.Compose([
    UnNormalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    torchvision.transforms.ToPILImage(),
    lambda x: np.asarray(x)
])
for pair, label in train_dataset:
    first_img, second_img = torch.unbind(pair, dim=0)
    first_img, second_img = un_normaliser(first_img), un_normaliser(second_img)
    if label[0] == 1:
        image = np.concatenate([first_img, second_img], axis=1)
    elif label[2] == 1:
        image = np.concatenate([second_img, first_img], axis=1)
    elif label[3] == 1:
        image = np.concatenate([second_img, first_img], axis=0)
    elif label[1] == 1:
        image = np.concatenate([first_img, second_img], axis=0)

    else:
        image = np.concatenate([first_img, np.zeros_like(first_img), second_img], axis=0)

    # image = cv2.bitwise_not(image)
    cv2.imshow('image', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    # waitKey() waits for a key press to close the window and 0 specifies indefinite loop
    cv2.waitKey(5000)

    # cv2.destroyAllWindows() simply destroys all the windows we created.
cv2.destroyAllWindows()
