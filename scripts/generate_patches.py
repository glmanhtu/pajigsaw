import argparse
import math
import os

import tqdm
from PIL import Image

parser = argparse.ArgumentParser('Pajigsaw patch generating script', add_help=False)
parser.add_argument('--data-path', required=True, type=str, help='path to dataset')
parser.add_argument('--output-path', required=True, type=str, help='path to output dataset')
parser.add_argument('--patch-size', type=int, default=128)
parser.add_argument('--erosion', type=float, default=0.07)
args = parser.parse_args()

patch_size = args.patch_size
gap = patch_size * args.erosion
images = []
for root, dirs, files in os.walk(args.data_path):
    for file in files:
        if file.lower().endswith((".jpg", ".png")):
            images.append(os.path.join(root, file))


fragment_map = {}
for image_path in tqdm.tqdm(images):
    with Image.open(image_path) as f:
        image = f.convert('RGB')

    # Resize the image if it does not fit the patch size that we want
    ratio = (patch_size * 4 + gap * 3) / min(image.width, image.height)
    if ratio > 1:
        image = image.resize((math.ceil(ratio * image.width), math.ceil(ratio * image.height)), Image.LANCZOS)

    group_patch_size = int(patch_size * 2 + gap), int(patch_size * 3 + gap * 2)

    image_name = os.path.splitext(os.path.basename(image_path))[0]
    patch_dir = os.path.join(args.output_path, image_name)
    os.makedirs(patch_dir, exist_ok=True)
    i = 0.
    while (i+1) * group_patch_size[0] <= image.height:
        j = 0
        while (j+1) * group_patch_size[1] <= image.width:
            box = (int(j*group_patch_size[1]), int(i*group_patch_size[0]),
                   int((j+1)*group_patch_size[1]), int((i+1)*group_patch_size[0]))
            patch = image.crop(box)
            patch_name = f'{i}_{j}.jpg'
            patch.save(os.path.join(patch_dir, patch_name))
            j += 0.5
        i += 0.5
