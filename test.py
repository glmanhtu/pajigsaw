import argparse
import datetime
import json
import os
import random
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from sklearn.metrics import accuracy_score, roc_auc_score
from timm.utils import AverageMeter

from config import get_config
from data.build import build_test_loader
from logger import create_logger
from models import build_model
from utils import load_pretrained


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
    parser.add_argument('--pretrained',
                        help='pretrained weight from checkpoint, could be imagenet22k pretrained weight')
    parser.add_argument('--disable_amp', action='store_true', help='Disable pytorch amp')
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--tag', help='tag of experiment')

    args, unparsed = parser.parse_known_args()

    config = get_config(args)

    return args, config


def main(config):
    dataset_test, data_loader_test = build_test_loader(config)

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
    loss, acc, auc = testing(config, data_loader_test, model)
    logger.info(f"Evaluation: AUC: {auc:.2f}% ACC: {acc:.2f}%, Loss: {loss}")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))


@torch.no_grad()
def testing(config, data_loader, model):
    criterion = torch.nn.BCEWithLogitsLoss()
    model.eval()

    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    auc_meter = AverageMeter()

    end = time.time()
    for idx, (images, target) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True).view(-1, 1)

        # compute output
        with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
            output = model(images)
        loss = criterion(output, target)
        y = target.cpu().numpy()
        y_hat = (output > 0).float().cpu().numpy()  # sigmoid 0 = 0.5
        acc = accuracy_score(y, y_hat) * 100
        auc = roc_auc_score(y, y_hat) * 100

        loss_meter.update(loss.item(), target.size(0))
        acc_meter.update(acc, target.size(0))
        auc_meter.update(auc, target.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            logger.info(
                f'Test: [{idx}/{len(data_loader)}]\t'
                f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                f'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'ACC {acc_meter.val:.3f} ({acc_meter.avg:.3f})\t'
                f'AUC {auc_meter.val:.3f} ({auc_meter.avg:.3f})\t'
                f'Mem {memory_used:.0f}MB')

    return loss_meter.avg, acc_meter.avg, auc_meter.avg


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
