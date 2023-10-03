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
from data.datasets.hisfrag20_test import HisFrag20Test
from misc.logger import create_logger
from misc.sampler import DistributedEvalSampler
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
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.DATA.BATCH_SIZE,
        shuffle=False,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=True,
        drop_last=False
    )

    indicates = torch.arange(len(dataset)).type(torch.int).cuda()
    pairs = torch.combinations(indicates, r=2)

    predicts = torch.zeros((0, 1), dtype=torch.float16).cuda()
    pair_indexes = torch.zeros((0, 2), dtype=torch.int32).cuda()
    start = time.time()
    end = time.time()
    batch_time = AverageMeter()
    for idx, (images, indexes) in enumerate(dataloader):
        images = images.cuda(non_blocking=True)
        lower_bound, higher_bound = torch.min(indexes), torch.max(indexes)
        mask_pairs = torch.logical_and(
            torch.all(torch.less_equal(pairs, higher_bound), dim=1),
            torch.all(torch.greater_equal(pairs, lower_bound), dim=1)
        )
        batch_pairs = pairs[mask_pairs]
        assert len(batch_pairs) > 0
        x1_indexes = batch_pairs[:, 0] - lower_bound
        x2_indexes = batch_pairs[:, 1] - lower_bound

        # compute output
        with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
            output = model(images[x1_indexes], images[x2_indexes])
        predicts = torch.cat([predicts, output])
        pair_indexes = torch.cat([pair_indexes, batch_pairs])

        batch_time.update(time.time() - end)
        end = time.time()
        if idx % config.PRINT_FREQ == 0:
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (len(dataloader) - idx)
            logger.info(
                f'Testing: [{idx}/{len(dataloader)}]]\t'
                f'eta {datetime.timedelta(seconds=int(etas))}\t'
                f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'mem {memory_used:.0f}MB')

    similarity_map = {}
    predicts = torch.sigmoid(predicts).cpu()
    for pred, index in zip(predicts.numpy(), pair_indexes.numpy()):
        img_1 = os.path.splitext(os.path.basename(dataset.samples[index[0]]))[0]
        img_2 = os.path.splitext(os.path.basename(dataset.samples[index[1]]))[0]
        similarity_map.setdefault(img_1, {})[img_2] = pred
        similarity_map.setdefault(img_2, {})[img_1] = pred

    result_file = os.path.join(config.OUTPUT, f'similarity_matrix.pkl')
    with open(result_file, 'wb') as f:
        pickle.dump(similarity_map, f, protocol=pickle.HIGHEST_PROTOCOL)

    epoch_time = time.time() - start
    logger.info(f"Testing takes {datetime.timedelta(seconds=int(epoch_time))}")


if __name__ == '__main__':
    args, config = parse_option()

    seed = config.SEED
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    os.makedirs(config.OUTPUT, exist_ok=True)
    logger = create_logger(output_dir=config.OUTPUT, dist_rank=0, name=f"{config.MODEL.NAME}", affix="_test")

    # print config
    logger.info(config.dump())
    logger.info(json.dumps(vars(args)))

    main(config)
