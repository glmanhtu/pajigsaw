import argparse
import glob
import os

import torchvision.transforms
import tqdm
from PIL import Image, ImageOps

from pajigsaw.data.fragment import fragment_image


def get_args_parser(add_help: bool = True):
    parser = argparse.ArgumentParser("Pajigsaw dataset generator", add_help=add_help)
    parser.add_argument("--dataset-dir", required=True, metavar="FILE", help="Path to papyrus images dataset")
    parser.add_argument("--patch-size", type=int, default=384, help="Size of the path fragment")
    parser.add_argument(
        "--output-dir",
        required=True,
        type=str,
        help="Output Pajigsaw dataset",
    )

    return parser


if __name__ == '__main__':
    args = get_args_parser().parse_args()
    splits = ['val_images', 'train_images', 'test_images']
    for split in splits:
        print(f'Processing split {split}')
        image_paths = glob.glob(os.path.join(args.dataset_dir, split, '**', '*.JPEG'), recursive=True)
        split_short = split.replace('_images', '')
        for image_path in tqdm.tqdm(image_paths):
            with Image.open(image_path) as f:
                img = f.convert('RGB')
                ratio = 2 * args.patch_size / min(img.width, img.height)
                img = img.resize((int(ratio * img.width), int(ratio * img.height)), Image.LANCZOS)
            n_rows = max(round(img.height / args.patch_size), 1)
            n_cols = max(round(img.width / args.patch_size), 1)
            patches, im_cut = fragment_image(img, n_cols, n_rows)
            file_name = os.path.splitext(os.path.basename(image_path))[0]
            if len(patches) < 3:
                continue
            os.makedirs(os.path.join(args.output_dir, split_short, file_name), exist_ok=True)
            for item in patches:
                patch = item['img']
                shape_trans = torchvision.transforms.Compose([
                    lambda x: ImageOps.invert(x),
                    torchvision.transforms.Pad(max(patch.size) // 3),
                    torchvision.transforms.RandomAffine(2, fill=0)
                ])
                patch = shape_trans(patch)
                bbox = patch.getbbox()
                patch = patch.crop(bbox)
                patch = ImageOps.invert(patch)
                patch_name = f'{item["col"]}_{item["row"]}.png'
                patch.save(os.path.join(args.output_dir, split_short, file_name, patch_name))
