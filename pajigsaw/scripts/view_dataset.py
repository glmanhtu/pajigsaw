import argparse

import matplotlib
import numpy as np
import torch
import torchvision.transforms
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from pajigsaw.data.datasets.papy_jigsaw import PapyJigSaw
from pajigsaw.data.transforms import TwoImgSyncAugmentation, UnNormalize, TwoImgSyncEval

matplotlib.use('MacOSX')

parser = argparse.ArgumentParser('Pajigsaw', add_help=False)
parser.add_argument('--image_size', default=224, type=int, help='Image size')
parser.add_argument('--data_path', default='/path/to/imagenet/train/', type=str,
                    help='Please specify path to the ImageNet training data.')

args = parser.parse_args()

un_normalize = torchvision.transforms.Compose([
    UnNormalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    torchvision.transforms.ToPILImage()
])
transform = TwoImgSyncEval(args.image_size)
dataset = PapyJigSaw(args.data_path, PapyJigSaw.Split.TRAIN, transform=transform)
dataloader = DataLoader(dataset, shuffle=True, batch_size=1)
for images, label in dataloader:
    print(label[0])
    x1, x2 = torch.unbind(images[0], dim=0)
    x1 = np.array(un_normalize(x1))
    x2 = np.array(un_normalize(x2))
    vis = np.concatenate((x1, x2), axis=1)

    fig = plt.figure()

    ax = fig.add_axes([0, 0, 1, 1])

    # Hide spines, ticks, etc.
    ax.axis('off')

    # Display the image.
    ax.imshow(vis, cmap='gray')
    plt.show()

