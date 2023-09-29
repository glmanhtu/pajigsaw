import argparse
import math
import os

import torchvision
import tqdm
from PIL import Image

from data import transforms

parser = argparse.ArgumentParser('Pajigsaw patch generating script', add_help=False)
parser.add_argument('--image-path', required=True, type=str, help='path to dataset')
parser.add_argument('--output-path', required=True, type=str, help='path to output dataset')
parser.add_argument('--patch-size', type=int, default=64)
parser.add_argument('--erosion', type=float, default=0.07)
args = parser.parse_args()

patch_size = args.patch_size
gap = patch_size * args.erosion
with Image.open(args.image_path) as f:
    image = f.convert('RGB')


cropper = torchvision.transforms.CenterCrop((patch_size * 2, patch_size * 3))
patch = cropper(image)

# Crop the image into a grid of 3 x 2 patches
crops = transforms.crop(patch, 3, 2)
erosion_ratio = args.erosion
piece_size_erosion = math.ceil(patch_size * (1 - erosion_ratio))
cropper = torchvision.transforms.CenterCrop(piece_size_erosion)

img_name = os.path.splitext(os.path.basename(args.image_path))[0]
os.makedirs(os.path.join(args.output_path, img_name), exist_ok=True)
for idx, crop in enumerate(crops):
    img = cropper(crop)
    patch_name = f'{idx}.jpg'
    img.save(os.path.join(args.output_path, img_name, patch_name))
