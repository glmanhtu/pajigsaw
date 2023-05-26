import argparse
import colorsys
import logging
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


def apply_mask(image, mask, color, alpha=0.5):
    for c in range(3):
        image[:, :, c] = image[:, :, c] * (1 - alpha * mask) + alpha * mask * color[c] * 255
    return image


def random_colors(N, bright=True):
    """
    Generate random colors.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors


def display_instances(image, mask, fname="test", figsize=(5, 5), blur=False, contour=True, alpha=0.5):
    fig = plt.figure(figsize=figsize, frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax = plt.gca()

    N = 1
    mask = mask[None, :, :]
    # Generate random colors
    colors = random_colors(N)

    # Show area outside image boundaries.
    height, width = image.shape[:2]
    margin = 0
    ax.set_ylim(height + margin, -margin)
    ax.set_xlim(-margin, width + margin)
    ax.axis('off')
    masked_image = image.astype(np.uint32).copy()
    for i in range(N):
        color = colors[i]
        _mask = mask[i]
        if blur:
            _mask = cv2.blur(_mask,(10,10))
        # Mask
        masked_image = apply_mask(masked_image, _mask, color, alpha)
        # Mask Polygon
        # Pad to ensure proper polygons for masks that touch image edges.
        if contour:
            padded_mask = np.zeros((_mask.shape[0] + 2, _mask.shape[1] + 2))
            padded_mask[1:-1, 1:-1] = _mask
            contours = find_contours(padded_mask, 0.5)
            for verts in contours:
                # Subtract the padding and flip (y, x) to (x, y)
                verts = np.fliplr(verts) - 1
                p = Polygon(verts, facecolor="none", edgecolor=color)
                ax.add_patch(p)
    ax.imshow(masked_image.astype(np.uint8), aspect='auto')
    fig.savefig(fname)
    return


def generate_attention_images(attentions, threshold, patch_size, w, h):
    nh = attentions.shape[1]  # number of head

    attentions = attentions[0, :, 0, 1:].reshape(nh, -1)

    # we keep only a certain percentage of the mass
    val, idx = torch.sort(attentions)
    val /= torch.sum(val, dim=1, keepdim=True)
    cumval = torch.cumsum(val, dim=1)
    th_attn = cumval > (1 - threshold)
    idx2 = torch.argsort(idx)
    for head in range(nh):
        th_attn[head] = th_attn[head][idx2[head]]
    th_attn = th_attn.reshape(nh, w, h).float()

    # interpolate
    th_attn = F.interpolate(th_attn.unsqueeze(0), scale_factor=patch_size, mode="nearest")[0].cpu().numpy()
    attentions = attentions.reshape(nh, w, h)
    attentions = F.interpolate(attentions.unsqueeze(0), scale_factor=patch_size, mode="nearest")[0].cpu().numpy()

    return th_attn, attentions


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

    torchvision.utils.save_image(torchvision.utils.make_grid(second, normalize=True, scale_each=True),
                                 os.path.join(args.output_dir, "img.png"))

    th_attn, attentions = generate_attention_images(x2_attn, args.threshold, config.MODEL.PJS.PATCH_SIZE,
                                                    w_featmap, h_featmap)

    fname = os.path.join(args.output_dir, "attn-2.jpg")
    plt.imsave(
        fname=fname,
        arr=sum(
            attentions[i] * 1 / attentions.shape[0]
            for i in range(attentions.shape[0])
        ),
        cmap="inferno",
        format="jpg",
    )


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
