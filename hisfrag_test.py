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
import torch.distributed as dist
import tqdm
from torch.utils.data import Dataset

from misc import wi19_evaluate
from config import get_config
from data.datasets.hisfrag20_test import HisFrag20Test
from data.transforms import TwoImgSyncEval
from misc.logger import create_logger
from models import build_model
from misc.utils import load_pretrained


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
    model_without_ddp = model

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], broadcast_buffers=False)

    if os.path.isfile(config.MODEL.PRETRAINED):
        load_pretrained(config, model_without_ddp, logger)
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
    dataset = HisFrag20Test(config.DATA.DATA_PATH, image_size=config.DATA.IMG_SIZE,
                            transform=TwoImgSyncEval(config.DATA.IMG_SIZE))
    sampler_val = torch.utils.data.distributed.DistributedSampler(
        dataset, shuffle=config.TEST.SHUFFLE
    )
    data_loader = torch.utils.data.DataLoader(
        dataset, sampler=sampler_val,
        batch_size=config.DATA.TEST_BATCH_SIZE,
        shuffle=False,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=False
    )

    distance_map = {}
    for images, targets in tqdm.tqdm(data_loader):
        images = images.cuda(non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
            output = model(images)

        for pred, entry_id in zip(torch.sigmoid(output).cpu().numpy(), targets.numpy()):
            i, j = entry_id
            distance_map.setdefault(i, {})[j] = pred

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
    local_rank = int(os.environ["LOCAL_RANK"])

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank = -1
        world_size = -1
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    torch.distributed.barrier()

    seed = config.SEED + dist.get_rank()
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    os.makedirs(config.OUTPUT, exist_ok=True)
    logger = create_logger(output_dir=config.OUTPUT, dist_rank=dist.get_rank(), name=f"{config.MODEL.NAME}",
                           affix="_test")

    # print config
    logger.info(config.dump())
    logger.info(json.dumps(vars(args)))

    main(config)
