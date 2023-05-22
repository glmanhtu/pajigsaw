import argparse
import glob
import os

import torch
import torchvision.transforms
import tqdm
from PIL import Image, ImageOps
from torch.utils.data import Dataset, DataLoader

from pajigsaw.data.fragment import fragment_image


def get_args_parser(add_help: bool = True):
    parser = argparse.ArgumentParser("Pajigsaw dataset generator", add_help=add_help)
    parser.add_argument("--dataset-dir", required=True, metavar="FILE", help="Path to papyrus images dataset")
    parser.add_argument("--n-workers", type=int, default=2, help="Number of workers")
    parser.add_argument("--patch-size", type=int, default=384, help="Size of the path fragment")
    parser.add_argument(
        "--output-dir",
        required=True,
        type=str,
        help="Output Pajigsaw dataset",
    )

    return parser


class ImageNetData(Dataset):
    def __init__(self, dataset_dir, part, output_dir, patch_size):
        self.image_paths = sorted(glob.glob(os.path.join(dataset_dir, part, '**', '*.JPEG'), recursive=True))
        self.working_dir = os.path.join(output_dir, part.replace('_images', ''))
        self.patch_size = patch_size

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        with Image.open(image_path) as f:
            img = f.convert('RGB')
            ratio = 2 * self.patch_size / min(img.width, img.height)
            img = img.resize((int(ratio * img.width), int(ratio * img.height)), Image.LANCZOS)
        n_rows = max(round(img.height / self.patch_size), 1)
        n_cols = max(round(img.width / self.patch_size), 1)
        patches, im_cut = fragment_image(img, n_cols, n_rows)
        file_name = os.path.splitext(os.path.basename(image_path))[0]
        if len(patches) < 3:
            return torch.tensor(0)
        os.makedirs(os.path.join(self.working_dir, file_name), exist_ok=True)
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
            patch.save(os.path.join(self.working_dir, file_name, patch_name))
        return torch.tensor(len(patches))


if __name__ == '__main__':
    args = get_args_parser().parse_args()
    splits = ['val_images', 'train_images', 'test_images']
    for split in splits:
        print(f'Processing split {split}')
        dataset = ImageNetData(args.dataset_dir, split, args.output_dir, args.patch_size)
        dataloader = DataLoader(dataset, batch_size=args.n_workers, num_workers=args.n_workers)
        pbar = tqdm.tqdm(dataloader)
        total_patches = 0
        for data in tqdm.tqdm(dataloader):
            total_patches += torch.sum(data).item()
            pbar.set_postfix({'total patches': total_patches})

