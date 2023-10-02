import argparse
import datetime
import json
import os
import pickle
import random
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torchvision
from timm.utils import AverageMeter
from torch.utils.data import Dataset, ConcatDataset

from config import get_config
from data.datasets.hisfrag20_test import HisFrag20Test, HisFrag20X1
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
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    x1_dataset = HisFrag20Test(config.DATA.DATA_PATH, transform=transform)
    x1_sampler = torch.utils.data.distributed.DistributedSampler(
        x1_dataset, shuffle=config.TEST.SHUFFLE
    )
    x1_dataloader = torch.utils.data.DataLoader(
        x1_dataset, sampler=x1_sampler,
        batch_size=config.DATA.BATCH_SIZE,
        shuffle=False,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=False
    )

    predicts = torch.zeros((0, 1), dtype=torch.float16).cuda()
    indexes = torch.zeros((0, 2), dtype=torch.int32)
    batch_time = AverageMeter()
    start = time.time()
    end = time.time()

    for x1_idx, (x1_images, x1_indexes) in enumerate(x1_dataloader):
        x1_images = x1_images.cuda(non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
            features_x1 = model(x1_images, forward_first_part=True)

        sub_datasets = []
        for x1_features, id1 in zip(features_x1.cpu(), x1_indexes):
            x1_offset = id1.item()
            x2_dataset = HisFrag20X1(config.DATA.DATA_PATH, transform=transform, samples=x1_dataset.samples[x1_offset:],
                                     x1_features=x1_features, x1_offset=x1_offset)
            sub_datasets.append(x2_dataset)

        x2_dataset = ConcatDataset(sub_datasets)
        sampler_val = torch.utils.data.distributed.DistributedSampler(
            x2_dataset, shuffle=config.TEST.SHUFFLE
        )
        x2_dataloader = torch.utils.data.DataLoader(
            x2_dataset, sampler=sampler_val,
            batch_size=config.DATA.BATCH_SIZE,
            shuffle=False,
            num_workers=config.DATA.NUM_WORKERS,
            pin_memory=config.DATA.PIN_MEMORY,
            drop_last=False
        )

        for x2_idx, (x2_images, x2_indexes, x1_features, x1_id) in enumerate(x2_dataloader):
            x2_images = x2_images.cuda(non_blocking=True)
            x1_features = x1_features.cuda(non_blocking=True)

            with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
                output = model(x2_images, x1_features)
            predicts = torch.cat([predicts, output])
            indexes = torch.cat([indexes, torch.column_stack([x1_id.expand(x2_indexes.shape[0]), x2_indexes + x1_id])])
            batch_time.update(time.time() - end)
            end = time.time()

            if x2_idx % config.PRINT_FREQ == 0:
                memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
                etas = batch_time.avg * (len(x2_dataloader) * len(x1_dataloader) - x1_idx * len(x1_dataloader) - x2_idx)
                logger.info(
                    f'Testing: [{x1_idx}/{len(x1_dataloader)}][{x2_idx}/{len(x2_dataloader)}]\t'
                    f'eta {datetime.timedelta(seconds=int(etas))}\t'
                    f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                    f'mem {memory_used:.0f}MB')

    similarity_map = {}
    predicts = torch.sigmoid(predicts).cpu()
    for pred, index in zip(predicts.numpy(), indexes.numpy()):
        img_1 = os.path.splitext(os.path.basename(x1_dataset.samples[index[0]]))[0]
        img_2 = os.path.splitext(os.path.basename(x1_dataset.samples[index[1]]))[0]
        similarity_map.setdefault(img_1, {})[img_2] = pred
        similarity_map.setdefault(img_2, {})[img_1] = pred

    result_file = os.path.join(config.OUTPUT, f'similarity_matrix_rank_{rank}.pkl')
    with open(result_file, 'wb') as f:
        pickle.dump(similarity_map, f, protocol=pickle.HIGHEST_PROTOCOL)

    # matrix = pd.DataFrame.from_dict(similarity_map, orient='index').sort_index()
    # matrix = matrix.reindex(sorted(matrix.columns), axis=1)
    #
    # matrix.to_csv(os.path.join(config.OUTPUT, 'similarity_matrix.csv'))
    # m_ap, top1, pr_a_k10, pr_a_k100 = wi19_evaluate.get_metrics(matrix, lambda x: x.split("_")[0])
    #
    # logger.info(
    #     f'mAP {m_ap:.3f}\t'
    #     f'Top 1 {top1:.3f}\t'
    #     f'Pr@k10 {pr_a_k10:.3f}\t'
    #     f'Pr@k100 {pr_a_k100:.3f}')

    epoch_time = time.time() - start
    logger.info(f"Testing takes {datetime.timedelta(seconds=int(epoch_time))}")


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
