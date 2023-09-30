import argparse
import os
import numpy as np
import torchvision
import tqdm

from data.datasets.hisfrag20_test import HisFrag20Test
from data.transforms import make_square

parser = argparse.ArgumentParser('Pajigsaw patch generating script', add_help=False)
parser.add_argument('--data-path', required=True, type=str, help='path to dataset')
parser.add_argument('--output-path', required=True, type=str, help='path to output dataset')
parser.add_argument('--image-size', type=int, default=512)
args = parser.parse_args()

transform = torchvision.transforms.Compose([
    lambda x: make_square(x),
    torchvision.transforms.Resize(size=args.image_size)
])
dataset = HisFrag20Test(args.data_path, transform=transform)
os.makedirs(args.output_path, exist_ok=True)
for image, index in tqdm.tqdm(dataset):
    file_path = dataset.samples[index.item()]
    file_name = os.path.basename(file_path)
    image.save(os.path.join(args.output_path, file_name))