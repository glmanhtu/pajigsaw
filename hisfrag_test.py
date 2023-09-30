import argparse
import datetime
import json
import os
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torchvision
import tqdm
from torch.utils.data import Dataset

from config import get_config
from data.datasets.hisfrag20_test import HisFrag20Test
from misc import wi19_evaluate
from misc.logger import create_logger
from misc.utils import load_pretrained
from models import build_model


def parse_option():
    parser = argparse.ArgumentParser('Pajigsaw testing script', add_help=False)
    parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    # easy config modification
    parser.add_argument('--batch-size', type=int, help="batch size for single GPU")
    parser.add_argument('--data-path', type=str, help='path to dataset')
    parser.add_argument('--pretrained', required=True, help='pretrained weight from checkpoint')
    parser.add_argument('--disable_amp', action='store_true', help='Disable pytorch amp')
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--tag', help='tag of experiment')

    args, unparsed = parser.parse_known_args()

    config = get_config(args)

    return args, config


def main(config):
    logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")
    model = build_model(config)
    logger.info(str(model))

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"number of params: {n_parameters}")

    model.cuda()

    if os.path.isfile(config.MODEL.PRETRAINED):
        load_pretrained(config, model, logger)
    else:
        raise Exception(f'Pretrained model is not exists {config.MODEL.PRETRAINED}')

    logger.info("Start testing")
    start_time = time.time()
    testing(config, model)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Test time {}'.format(total_time_str))


@torch.no_grad()
def testing(config, model):
    model.eval()
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    dataset = HisFrag20Test(config.DATA.DATA_PATH, transform=transform)

    predicts = torch.zeros((0, 1), dtype=torch.float16).cuda()
    indexes = torch.zeros((0, 2), dtype=torch.int32)
    pbar = tqdm.tqdm(dataset)
    for image, index in pbar:
        x1_id = index.item()
        image = image.cuda(non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
            x1 = model.forward_first_part(image[None])

        sub_dataset = HisFrag20Test(config.DATA.DATA_PATH,
                                    transform=transform, samples=dataset.samples[x1_id:])
        data_loader = torch.utils.data.DataLoader(
            sub_dataset,
            batch_size=config.DATA.BATCH_SIZE,
            shuffle=False,
            num_workers=config.DATA.NUM_WORKERS,
            pin_memory=config.DATA.PIN_MEMORY,
            drop_last=False
        )

        count = 0
        for images, x2_indexes in data_loader:
            images = images.cuda(non_blocking=True)

            with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
                x2 = model.forward_second_part(x1.expand(images.shape[0], -1, -1), images)
                output = model.forward_head(x2)
            predicts = torch.cat([predicts, output])
            indexes = torch.cat([indexes, torch.column_stack([index.expand(x2_indexes.shape[0]), x2_indexes + x1_id])])
            count += 1
            pbar.set_description(f"Processing {count}/{len(data_loader)}")

    distance_map = {}
    for pred, index in zip(torch.sigmoid(predicts).cpu().numpy(), indexes.numpy()):
        distance_map.setdefault(index[0], {})[index[1]] = pred
        distance_map.setdefault(index[1], {})[index[0]] = pred
    matrix = pd.DataFrame.from_dict(distance_map, orient='index').sort_index()
    matrix = matrix.reindex(sorted(matrix.columns), axis=1)
    m_ap, top1, pr_a_k10, pr_a_k100 = wi19_evaluate.get_metrics(matrix, dataset.get_group_id)

    logger.info(
        f'mAP {m_ap:.3f}\t'
        f'Top 1 {top1:.3f}\t'
        f'Pr@k10 {pr_a_k10:.3f}\t'
        f'Pr@k100 {pr_a_k100:.3f}')


if __name__ == '__main__':
    args, config = parse_option()
    seed = config.SEED
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    os.makedirs(config.OUTPUT, exist_ok=True)
    logger = create_logger(output_dir=config.OUTPUT, name=f"{config.MODEL.NAME}", affix="_test")

    # print config
    logger.info(config.dump())
    logger.info(json.dumps(vars(args)))

    main(config)
