import argparse
import colorsys
import logging
import math
import os
import random

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torchvision
from PIL import Image
import torch.nn.functional as F
from matplotlib import pyplot as plt
from matplotlib.patches import Polygon
from skimage.measure import find_contours

from config import get_config
from data.transforms import TwoImgSyncEval
from models import build_model
from utils import load_pretrained


def parse_option():
    parser = argparse.ArgumentParser('Pajigsaw visualising script', add_help=False)
    parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    # easy config modification
    parser.add_argument('--pretrained', required=True, help='pretrained weight from checkpoint')
    parser.add_argument('--images', type=str, required=True, nargs='+', help='Path to the two testing images')
    parser.add_argument('--output_dir', type=str, default='visualisation', help='Visualisation output dir')
    parser.add_argument('--disable_amp', action='store_true', help='Disable pytorch amp')
    parser.add_argument("--threshold", type=float, default=0.6, help="""We visualize masks
            obtained by thresholding the self-attention maps to keep xx% of the mass.""")
    parser.add_argument('--tag', help='tag of experiment')

    args, unparsed = parser.parse_known_args()
    args.keep_attn = True

    return args, get_config(args)


def random_colors(N, bright=True):
    """
    Generate random colors.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors


def main(config):
    model = build_model(config).cuda()

    if os.path.isfile(config.MODEL.PRETRAINED):
        load_pretrained(config, model, logger)
    else:
        raise Exception(f'Pretrained model is not exists {config.MODEL.PRETRAINED}')

    transform = TwoImgSyncEval(config.DATA.IMG_SIZE)
    assert len(args.images) == 2
    images = []
    for img_path in args.images:
        with Image.open(img_path) as f:
            images.append(f.convert('RGB'))
    first, second = transform(images[0], images[1])
    images = torch.stack([first, second], dim=0).cuda()
    with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
        yhat, x1_attn, x2_attn, cross_attn = model(images[None])

    w_featmap = images.shape[-2] // config.MODEL.PJS.PATCH_SIZE
    h_featmap = images.shape[-1] // config.MODEL.PJS.PATCH_SIZE

    nh = cross_attn.shape[1]  # number of head
    attentions = cross_attn[0, :, 1:, :]
    colours = random_colors(attentions.shape[1])

    for h in range(nh):
        attn_x1_img = np.zeros([w_featmap, h_featmap, 3], dtype=np.float32)
        attn_x2_img = np.zeros([w_featmap, h_featmap, 3], dtype=np.float32)

        for x2_feat_point in range(attentions.shape[1]):
            attention_x1 = attentions[h, x2_feat_point, :].reshape(w_featmap, h_featmap)
            if not torch.all(torch.le(attention_x1, args.threshold)):
                colour = colours[x2_feat_point]
                row = int(x2_feat_point / w_featmap)
                col = x2_feat_point - row * w_featmap
                attn_x2_img[row][col] = colour
                attn_x1_candidates = torch.gt(attention_x1, args.threshold)
                attn_x1_img[attn_x1_candidates.cpu().numpy()] = colour



    attention_x1 = torch.nn.functional.interpolate(attention_x1.unsqueeze(
        0), scale_factor=config.MODEL.PJS.PATCH_SIZE, mode="nearest")[0].cpu().numpy()


if __name__ == '__main__':
    args, config = parse_option()

    seed = config.SEED
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    os.makedirs(config.OUTPUT, exist_ok=True)
    logger = logging.getLogger(f"{config.MODEL.NAME}")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    main(config)
