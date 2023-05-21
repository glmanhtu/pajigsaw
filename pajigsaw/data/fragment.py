# Inspired from https://github.com/seuretm/diamond-square-fragmentation/blob/master/runme_generate_fragments.py
import os
import random

import numpy as np
from scipy.ndimage import label

import numpy
from PIL import Image, ImageChops
from torchvision import transforms


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
    ing_w = img.size[0]
    ing_h = img.size[1]

    # side borders
    b = horizon(arr.shape[0], 0.9, 10)
    for i in range(arr.shape[0]):
        arr[i, 0:b[i]] = 0
    b = horizon(arr.shape[0], 0.9, 10)
    for i in range(arr.shape[0]):
        arr[i, (arr.shape[1] - b[-i]):arr.shape[1]] = 0
    b = horizon(arr.shape[1], 0.9, 10)
    for i in range(arr.shape[1]):
        arr[0:b[i], i] = 0
    b = horizon(arr.shape[1], 0.9, 10)
    for i in range(arr.shape[1]):
        arr[(arr.shape[0] - b[-i]):arr.shape[0], i] = 0

    # vertical cuts
    patch_w = (ing_w / n_cols)
    for cut in range(1, n_cols):
        h1 = horizon(arr.shape[0], 0.6, patch_w // 3)
        h2 = horizon(arr.shape[0], 0.5, 10)
        h3 = horizon(arr.shape[0], 0.8, patch_w // 6)
        h = h1 + h2 + cut * arr.shape[1] // n_cols
        for i in range(arr.shape[0]):
            j = (i + 1) % arr.shape[0]
            a = min(h[i], h[j]) - h3[i] // 2
            b = max(h[i], h[j]) + h3[i] // 2 + 2
            arr[i, a:b] = 0

    # horizontal cuts
    patch_h = (ing_h / n_rows)
    for cut in range(1, n_rows):
        h1 = horizon(arr.shape[1], 0.6, patch_h // 3)
        h2 = horizon(arr.shape[1], 0.5, 10)
        h3 = horizon(arr.shape[1], 0.8, patch_h // 6)
        h = h1 + h2 + cut * arr.shape[0] // n_rows
        for i in range(arr.shape[1]):
            j = (i + 1) % arr.shape[1]
            a = min(h[i], h[j]) - h3[i] // 2
            b = max(h[i], h[j]) + h3[i] // 2 + 2
            arr[a:b, i] = 0
    os.makedirs('img_cuts', exist_ok=True)
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
