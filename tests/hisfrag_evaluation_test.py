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
import torchvision
from timm.utils import AverageMeter
from torch.utils.data import Dataset

from config import get_config
from data.datasets.hisfrag20_test import HisFrag20GT
from hisfrag_test import hisfrag_eval
from misc import wi19_evaluate
from misc.logger import create_logger
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
    parser.add_argument('--batch-size', type=int, help="batch size for data")
    parser.add_argument('--max-n-authors', type=int, default=50)
    parser.add_argument('--data-path', type=str, help='path to dataset')
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
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], broadcast_buffers=False)

    logger.info("Start testing")
    start_time = time.time()
    max_author = args.max_n_authors
    similarity_map = hisfrag_eval(config, model, max_author, logger=logger)
    similarity_map = pd.DataFrame.from_dict(similarity_map, orient='index').sort_index()
    similarity_map = similarity_map.reindex(sorted(similarity_map.columns), axis=1)
    similarity_map.to_csv('similarity_matrix.csv')
    logger.info('Starting to calculate performance...')
    m_ap, top1, pr_a_k10, pr_a_k100 = wi19_evaluate.get_metrics(similarity_map, lambda x: x.split("_")[0])
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info(f'New approach: mAP {m_ap:.3f}\t' f'Top 1 {top1:.3f}\t' f'Pr@k10 {pr_a_k10:.3f}\t' 
                f'Pr@k100 {pr_a_k100:.3f} Time: {total_time_str}')

    start_time = time.time()
    similarity_map = hisfrag_eval_original(config, model, max_authors=max_author)
    similarity_map = pd.DataFrame.from_dict(similarity_map, orient='index').sort_index()
    similarity_map = similarity_map.reindex(sorted(similarity_map.columns), axis=1)
    logger.info('Starting to calculate performance...')
    m_ap2, top1, pr_a_k10, pr_a_k100 = wi19_evaluate.get_metrics(similarity_map, lambda x: x.split("_")[0])
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info(f'Original approach: mAP {m_ap2:.3f}\t' f'Top 1 {top1:.3f}\t' f'Pr@k10 {pr_a_k10:.3f}\t'
                f'Pr@k100 {pr_a_k100:.3f} Time: {total_time_str}')
    similarity_map.to_csv('similarity_matrix_2.csv')

    logger.info(f'First: {m_ap}, second: {m_ap2}')
    assert m_ap == m_ap2


@torch.no_grad()
def hisfrag_eval_original(config, model, max_authors=None):
    model.eval()
    transform = torchvision.transforms.Compose([
        torchvision.transforms.CenterCrop(config.DATA.IMG_SIZE),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    dataset = HisFrag20GT(config.DATA.DATA_PATH, transform=transform, max_n_authors=max_authors)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.DATA.BATCH_SIZE,
        shuffle=False,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=True,
        drop_last=False
    )

    predicts = torch.zeros((0, 1), dtype=torch.float16).cuda()
    pair_indexes = torch.zeros((0, 2), dtype=torch.int32)

    batch_time = AverageMeter()
    end = time.time()
    for idx, (images, pair) in enumerate(dataloader):
        images = images.cuda(non_blocking=True)
        pair_indexes = torch.cat([pair_indexes, pair])
        with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
            output = model(images)

        predicts = torch.cat([predicts, output])
        batch_time.update(time.time() - end)
        end = time.time()
        if idx % config.PRINT_FREQ == 0:
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (len(dataloader) - idx)
            logger.info(
                f'Testing: [{idx}/{len(dataloader)}]\t'
                f'X2 eta {datetime.timedelta(seconds=int(etas))}\t'
                f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'mem {memory_used:.0f}MB')

    similarity_map = {}
    predicts = torch.sigmoid(predicts).cpu()
    for pred, index in zip(predicts.numpy(), pair_indexes.numpy()):
        img_1 = os.path.splitext(os.path.basename(dataset.samples[index[0]]))[0]
        img_2 = os.path.splitext(os.path.basename(dataset.samples[index[1]]))[0]
        similarity_map.setdefault(img_1, {})[img_2] = pred[0]
        similarity_map.setdefault(img_2, {})[img_1] = pred[0]
    return similarity_map


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
    logger = create_logger(output_dir=config.OUTPUT, dist_rank=dist.get_rank(),
                           name=f"{config.MODEL.NAME}", affix="_test")

    # print config
    logger.info(config.dump())
    logger.info(json.dumps(vars(args)))

    main(config)
