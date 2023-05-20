import argparse
import glob
import os

import matplotlib
import numpy as np
import torch
import torchvision.transforms
from PIL import Image, ImageOps
from matplotlib import pyplot as plt

from pajigsaw.data.datasets.papy_jigsaw import PapyJigSaw
from pajigsaw.data.transforms import TwoImgSyncAugmentation, UnNormalize, TwoImgSyncEval

matplotlib.use('MacOSX')

parser = argparse.ArgumentParser('Pajigsaw', add_help=False)
parser.add_argument('--data_path', default='/path/to/imagenet/train/', type=str,
                    help='Please specify path to the ImageNet training data.')

args = parser.parse_args()


def visualize_patches(image_patches, patch_locations, grid_shape):
    num_patches = len(image_patches)
    assert num_patches == len(patch_locations), "Number of patches and locations must be the same."
    assert grid_shape[0] * grid_shape[1] >= num_patches, "Grid shape is too small to accommodate all patches."

    patch_size = image_patches[0].shape[1:]  # Assuming all patches have the same size

    # Create a blank canvas for the image grid
    grid = torch.zeros((3, patch_size[0] * grid_shape[0], patch_size[1] * grid_shape[1]))

    # Populate the grid with image patches at their corresponding locations
    for patch_idx in range(num_patches):
        patch = image_patches[patch_idx]
        location = patch_locations[patch_idx]
        start_row = location[0] * patch_size[0]
        start_col = location[1] * patch_size[1]
        grid[:, start_row:start_row+patch_size[0], start_col:start_col+patch_size[1]] = patch

    # Convert the tensor grid to a numpy array
    grid_np = grid.numpy().transpose(1, 2, 0)

    # Visualize the image grid
    plt.imshow(grid_np)
    plt.axis('off')
    plt.show()


fragment_paths = glob.glob(os.path.join(args.data_path, '**', '*.png'), recursive=True)
fragment_map = {}
for fragment_path in fragment_paths:
    image_name = os.path.basename(os.path.dirname(fragment_path))
    fragment_name = os.path.splitext(os.path.basename(fragment_path))[0]
    row, col = map(int, fragment_name.split("_"))
    fragment_map.setdefault(image_name, []).append({'img': fragment_path, 'col': col, 'row': row,
                                                    'positive': [], 'negative': []})

for im_name, fragments in fragment_map.items():
    if '273r_2_1' not in im_name:
        continue
    print(im_name)
    n_cols = max([x['col'] for x in fragments]) + 1
    n_row = max([x['row'] for x in fragments]) + 1
    im_sizes = []
    for fragment in fragments:
        with Image.open(fragment['img']) as f:
            im = f.convert('RGB')
        im_sizes += [im.width, im.height]
    max_size = max(im_sizes)
    im_resizer = torchvision.transforms.Compose([
        lambda x: ImageOps.invert(x),
        torchvision.transforms.CenterCrop(max_size),
        lambda x: ImageOps.invert(x)
    ])
    image_patches = []
    patch_locations = []
    for fragment in fragments:
        with Image.open(fragment['img']) as f:
            im = f.convert('RGB')
        im = im_resizer(im)
        image_patches.append(torchvision.transforms.ToTensor()(im))
        patch_locations.append((fragment['col'], fragment['row']))

    grid_shape = (n_cols, n_row)
    visualize_patches(image_patches, patch_locations, grid_shape)

