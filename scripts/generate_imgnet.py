# Inspired from https://github.com/seuretm/diamond-square-fragmentation/blob/master/runme_generate_fragments.py

import argparse
import glob
import math
import os
import random

import numpy
import numpy as np
import torch
import tqdm
from PIL import Image, ImageChops
from datasets import load_dataset
from scipy.ndimage import label
from torch.utils.data import Dataset, DataLoader


def get_args_parser(add_help: bool = True):
    parser = argparse.ArgumentParser("Pajigsaw dataset generator", add_help=add_help)
    parser.add_argument("--dataset-dir", required=True, metavar="FILE", help="Path to papyrus images dataset")
    parser.add_argument("--n-workers", type=int, default=0, help="Number of workers")
    parser.add_argument("--patch-size", type=int, default=224, help="Size of the path fragment")
    parser.add_argument(
        "--output-dir",
        required=True,
        type=str,
        help="Output Pajigsaw dataset",
    )

    return parser


def horizon(length, roughness=1., amplitude=1.):
    res = numpy.zeros(length)
    res[0] = random.uniform(0, 1)
    res[-1] = random.uniform(0, 1)
    horizon_step(res, 0, length - 1, roughness, roughness)
    a = min(res)
    b = max(res)
    res = numpy.round((numpy.array(res) - a) / (b - a) * amplitude).astype(int)
    return res


def horizon_step(arr, a, c, noise, roughness):
    if c - a <= 1:
        return
    b = int((c - a) / 2 + .5) + a
    arr[b] = (arr[a] + arr[c]) / 2.0 + random.uniform(-noise / 2, noise / 2)
    horizon_step(arr, a, b, noise * roughness, roughness)
    horizon_step(arr, b, c, noise * roughness, roughness)


def avg_pixel_score(np_img):
    pixel_sum = np.sum(np_img)
    n_pixels = np_img.shape[0] * np_img.shape[1] * np_img.shape[2]
    return pixel_sum / n_pixels


def fragment_image(img: Image, n_cols: int, n_rows: int):
    """
    This function fragment the input image to multiple fragments, specified by n_cols and n_rows
    """
    arr = 255 * numpy.ones([img.size[1], img.size[0]])
    ing_w = img.width
    ing_h = img.height

    patch_w = (ing_w // (n_cols + 1))
    patch_h = (ing_h // (n_rows + 1))

    #  side borders
    # Left vertical border
    h1 = horizon(ing_h, 0.6, patch_w // 6)
    h2 = horizon(ing_h, 0.5, 10)
    h3 = horizon(ing_h, 0.8, patch_w // 18)
    h = h1 + h2
    for i in range(ing_h):
        j = (i + 1) % ing_h
        b = max(h[i], h[j]) + h3[i] // 2
        arr[i, :b] = 0

    # Right vertical border
    h1 = horizon(ing_h, 0.6, patch_w // 6)
    h2 = horizon(ing_h, 0.5, 10)
    h3 = horizon(ing_h, 0.8, patch_w // 12)
    h = h1 + h2 + (ing_w - 10 - patch_w // 6)
    for i in range(ing_h):
        j = (i + 1) % ing_h
        a = min(h[i], h[j]) - h3[i] // 2
        arr[i, a:] = 0

    h1 = horizon(ing_w, 0.6, patch_h // 6)
    h2 = horizon(ing_w, 0.5, 10)
    h3 = horizon(ing_w, 0.8, patch_h // 18)
    h = h1 + h2
    for i in range(ing_w):
        j = (i + 1) % ing_w
        b = max(h[i], h[j]) + h3[i] // 2
        arr[:b, i] = 0

    h1 = horizon(ing_w, 0.6, patch_h // 6)
    h2 = horizon(ing_w, 0.5, 10)
    h3 = horizon(ing_w, 0.8, patch_h // 12)
    h = h1 + h2 + (ing_h - 10 - patch_h // 6)

    for i in range(ing_w):
        j = (i + 1) % ing_w
        a = min(h[i], h[j]) - h3[i] // 2
        arr[a:, i] = 0

    patch_w = (ing_w // n_cols)
    patch_h = (ing_h // n_rows)

    # vertical cuts
    for cut in range(1, n_cols):
        h1 = horizon(ing_h, 0.6, patch_w // 6)
        h2 = horizon(ing_h, 0.5, 10)
        h3 = horizon(ing_h, 0.8, patch_w // 12)
        h = h1 + h2 + int(cut * 0.9 * ing_w // n_cols)
        for i in range(ing_h):
            j = (i + 1) % ing_h
            a = min(h[i], h[j]) - h3[i] // 2
            b = max(h[i], h[j]) + h3[i] // 2 + int(0.01 * ing_h)
            arr[i, a:b] = 0

    # horizontal cuts
    for cut in range(1, n_rows):
        h1 = horizon(ing_w, 0.6, patch_h // 6)
        h2 = horizon(ing_w, 0.5, 10)
        h3 = horizon(ing_w, 0.8, patch_h // 12)
        h = h1 + h2 + int(cut * 0.9 * ing_h // n_rows)
        for i in range(ing_w):
            j = (i + 1) % ing_w
            a = min(h[i], h[j]) - h3[i] // 2
            b = max(h[i], h[j]) + h3[i] // 2 + int(0.01 * ing_w)
            arr[a:b, i] = 0

    cuts = ImageChops.multiply(Image.fromarray(arr.astype('uint8')).convert('RGB'), img)

    labeled, ncomponents = label(arr)

    nb_pixels = (ing_w / n_cols) * (ing_h / n_rows)
    mx = numpy.max(labeled) + 1

    results = []
    for c in range(1, mx):
        binary = labeled == c
        s = numpy.sum(binary)
        if s < nb_pixels / 4:
            continue

        fmask = Image.fromarray((255 * binary).astype('uint8')).convert('RGB')
        bbox = fmask.getbbox()

        # Trying to find the correct coordinate of this fragment
        bbox_center = (bbox[2] + bbox[0]) / 2, (bbox[3] + bbox[1]) / 2
        vertical_pos = int(bbox_center[0] / (ing_w / n_cols))
        horizontal_pos = int(bbox_center[1] / (ing_h / n_rows))
        res = ImageChops.multiply(fmask, img).crop(bbox)
        binary = binary[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        data = numpy.array(res)
        data[np.logical_not(binary)] = 255
        pixel_score = avg_pixel_score(data)
        if pixel_score > 200:
            continue
        res = Image.fromarray(data)
        results.append({
            'row': horizontal_pos,
            'col': vertical_pos,
            'img': res
        })

    return results, cuts


class ImageNetData(Dataset):
    def __init__(self, dataset_dir, part, output_dir, patch_size):
        self.dataset = load_dataset("imagenet-1k", split=part, cache_dir=dataset_dir)
        # self.dataset = load_dataset("imagefolder", data_dir=dataset_dir)['validation']
        self.working_dir = os.path.join(output_dir, part)
        self.patch_size = patch_size

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_dir = os.path.join(self.working_dir, f'{index}')
        if os.path.exists(img_dir):
            n_patches = len(list(glob.glob(os.path.join(img_dir, '*.jpeg'))))
            return torch.tensor(n_patches)

        img = self.dataset[index]['image'].convert('RGB')
        ratio_short = self.patch_size * 1.2 / min(img.width, img.height)
        ratio_long = (self.patch_size * 2.2) / max(img.width, img.height)
        ratio = max(ratio_long, ratio_short)
        if ratio > 1:
            img = img.resize((math.ceil(ratio * img.width), math.ceil(ratio * img.height)), Image.LANCZOS)
        elif ratio < 0.25:
            ratio_short = self.patch_size * 4.2 / min(img.width, img.height)
            ratio_long = (self.patch_size * 4.2) / max(img.width, img.height)
            ratio = max(ratio_long, ratio_short)
            img = img.resize((math.ceil(ratio * img.width), math.ceil(ratio * img.height)), Image.LANCZOS)

        n_rows = max(round(img.height / self.patch_size), 1)
        n_cols = max(round(img.width / self.patch_size), 1)
        patches, im_cut = fragment_image(img, n_cols, n_rows)
        os.makedirs(os.path.join(self.working_dir, f'{index}'), exist_ok=True)
        for item in patches:
            patch = item['img']
            patch_name = f'{item["col"]}_{item["row"]}.jpeg'
            patch.save(os.path.join(self.working_dir, f'{index}', patch_name))
        return torch.tensor(len(patches))


if __name__ == '__main__':
    args = get_args_parser().parse_args()
    splits = ['validation', 'train', 'test']
    for split in splits:
        print(f'Processing split {split}')
        dataset = ImageNetData(args.dataset_dir, split, args.output_dir, args.patch_size)
        dataloader = DataLoader(dataset, batch_size=20, num_workers=args.n_workers)
        pbar = tqdm.tqdm(dataloader)
        total_patches = 0
        for data in tqdm.tqdm(dataloader):
            total_patches += torch.sum(data).item()
            pbar.set_postfix({'total patches': total_patches})
