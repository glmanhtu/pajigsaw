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
import torch.nn.functional as F
import torchvision
from timm.utils import AverageMeter
from torch.utils.data import Dataset

from config import get_config
from data.datasets.hisfrag20_test import HisFrag20Test
from misc import wi19_evaluate, utils
from misc.logger import create_logger
from data.samplers import DistributedOrderedIndicatesSampler
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
    parser.add_argument('--batch-size', type=int, help="batch size for data")
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
        raise Exception(f'Pretrained model doesn\'t exists {config.MODEL.PRETRAINED}')

    logger.info("Start testing")
    start_time = time.time()
    distance_matrix, img_names = hisfrag_eval_wrapper(config, model, None, world_size, rank, logger)
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Test time {}'.format(total_time_str))
    if rank == 0:
        df = pd.DataFrame(data=distance_matrix, columns=img_names, index=img_names)
        result_file = os.path.join(config.OUTPUT, f'distance_matrix_rank{rank}.csv')
        df.to_csv(result_file, index=True)


def hisfrag_eval_wrapper(config, model, max_authors=None, world_size=1, rank=0, logger=None):
    distance_matrix, img_names = hisfrag_eval(config, model, max_authors, world_size, rank, logger)
    logger.info('Starting to calculate performance...')

    # Get authors
    labels = utils.list_to_idx(img_names, lambda x: x.split('_')[0])
    m_ap, top1, pr_a_k10, pr_a_k100 = wi19_evaluate.get_metrics(distance_matrix, np.asarray(labels))

    logger.info(f'mAP {m_ap:.3f}\t' f'Top 1 {top1:.3f}\t' f'Pr@k10 {pr_a_k10:.3f}\t' f'Pr@k100 {pr_a_k100:.3f}')
    return distance_matrix, img_names


@torch.no_grad()
def hisfrag_eval(config, model, max_authors=None, world_size=1, rank=0, logger=None):
    model.eval()
    transform = torchvision.transforms.Compose([
        torchvision.transforms.CenterCrop(512),
        torchvision.transforms.Resize(config.DATA.IMG_SIZE),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    dataset = HisFrag20Test(config.DATA.DATA_PATH, transform=transform, max_n_authors=max_authors)
    print(f"Len dataset: {len(dataset)}")
    indicates = torch.arange(len(dataset)).type(torch.int).cuda()
    pairs = torch.combinations(indicates, r=2, with_replacement=True)
    del indicates

    sampler_val = DistributedOrderedIndicatesSampler(pairs[:, 0].cpu(), num_replicas=world_size, rank=rank)
    x1_dataloader = torch.utils.data.DataLoader(
        dataset, sampler=sampler_val,
        batch_size=config.DATA.BATCH_SIZE,
        shuffle=False,  # Very important, shuffle have to be False to maintain the order of the sample indexes
        num_workers=0,
        pin_memory=True,
        drop_last=False
    )
    predicts = torch.zeros((0, 3), dtype=torch.float32).cuda()
    tmp_data_path = os.path.join(config.OUTPUT, f'test_result_rank{rank}.pt')
    if not os.path.exists(tmp_data_path):
        batch_time = AverageMeter()
        for x1_idx, (x1, x1_indexes) in enumerate(x1_dataloader):
            x1_lower_bound, x1_upper_bound = x1_indexes[0], x1_indexes[-1]
            x1 = x1.cuda(non_blocking=True)
            pair_masks = torch.greater_equal(pairs[:, 0], x1_lower_bound)
            pair_masks = torch.logical_and(pair_masks, torch.less_equal(pairs[:, 0], x1_upper_bound))

            x2_dataset = HisFrag20Test(config.DATA.DATA_PATH, transform=transform, samples=dataset.samples,
                                       lower_bound=x1_lower_bound.item())
            logger.info(f'X2 dataset size: {len(x2_dataset)}, lower_bound: {x1_lower_bound}')
            x2_dataloader = torch.utils.data.DataLoader(
                x2_dataset,
                batch_size=config.DATA.BATCH_SIZE,
                shuffle=False,  # Very important, shuffle have to be False
                num_workers=config.DATA.NUM_WORKERS,
                pin_memory=True,
                drop_last=False
            )

            with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
                x1 = model(x1, forward_first_part=True)

            x1_pairs = pairs[pair_masks]
            end = time.time()
            for x2_id, (x2, x2_indicates) in enumerate(x2_dataloader):
                x2 = x2.cuda(non_blocking=True)
                x2_lower_bound, x2_upper_bound = x2_indicates[0], x2_indicates[-1]
                pair_masks = torch.greater_equal(x1_pairs[:, 1], x2_lower_bound)
                pair_masks = torch.logical_and(pair_masks, torch.less_equal(x1_pairs[:, 1], x2_upper_bound))
                x1_x2_pairs = x1_pairs[pair_masks]
                x1_pairs = x1_pairs[x1_pairs[:, 1] > x2_upper_bound]
                for sub_pairs in torch.split(x1_x2_pairs, config.DATA.TEST_BATCH_SIZE):
                    x1_sub = x1[sub_pairs[:, 0] - x1_lower_bound]
                    x2_sub = x2[sub_pairs[:, 1] - x2_lower_bound]
                    with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
                        outputs = model(x1_sub, x2_sub)
                    sub_pair_predicts = torch.column_stack([sub_pairs.float(), outputs.float()])
                    predicts = torch.cat([predicts, sub_pair_predicts])
                batch_time.update(time.time() - end)
                end = time.time()
                if x2_id % config.PRINT_FREQ == 0:
                    memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
                    etas = batch_time.avg * (len(x2_dataloader) - x2_id)
                    logger.info(
                        f'Testing: [{x1_idx}/{len(x1_dataloader)}][{x2_id}/{len(x2_dataloader)}]\t'
                        f'X2 eta {datetime.timedelta(seconds=int(etas))}\t'
                        f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                        f'mem {memory_used:.0f}MB')

        torch.save(predicts, tmp_data_path)

    for i in range(world_size):
        logger.info(f"Waiting for rank {i}.")
        logger.info("Note, this works only when all the ranks use the same file system! "
                    "Otherwise, this will hang forever...!")
        rank_data_path = os.path.join(config.OUTPUT, f'test_result_rank{i}.pt')
        while not os.path.exists(rank_data_path):
            time.sleep(120)

    logger.info(f"Gathering data...")
    predicts = []
    for i in range(world_size):
        rank_data_path = os.path.join(config.OUTPUT, f'test_result_rank{i}.pt')
        predicts.append(torch.load(rank_data_path, map_location='cpu'))

    predicts = torch.cat(predicts)

    logger.info(f"Generating similarity map...")
    assert len(predicts) == len(pairs)

    size = len(dataset.samples)

    # Initialize a similarity matrix with zeros
    similarity_matrix = torch.zeros((size, size), dtype=torch.float16)

    # Extract index pairs and scores
    indices = predicts[:, :2].long()
    scores = predicts[:, 2].type(torch.float16)

    # Use indexing and broadcasting to fill the similarity matrix
    similarity_matrix[indices[:, 0], indices[:, 1]] = scores
    similarity_matrix[indices[:, 1], indices[:, 0]] = scores

    logger.info(f"Converting to distance matrix...")
    # max_score = torch.amax(similarity_matrix, dim=1)
    distance_matrix = 1 - similarity_matrix

    labels = []
    for i in range(size):
        labels.append(os.path.splitext(os.path.basename(dataset.samples[i]))[0])
    logger.info("Distance matrix is generated!")
    return distance_matrix.numpy(), labels


if __name__ == '__main__':
    args, config = parse_option()
    local_rank, rank, world_size = -1, -1, -1

    if 'RANK' in os.environ:
        rank = int(os.environ["RANK"])

    if 'WORLD_SIZE' in os.environ:
        world_size = int(os.environ['WORLD_SIZE'])

    if 'LOCAL_RANK' in os.environ:  # for torch.distributed.launch
        local_rank = int(os.environ["LOCAL_RANK"])

    elif 'SLURM_PROCID' in os.environ:    # for slurm scheduler
        rank = int(os.environ['SLURM_PROCID'])
        local_rank = rank % torch.cuda.device_count()

    print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
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
