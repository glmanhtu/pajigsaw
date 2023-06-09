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
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score
from timm.utils import AverageMeter

import torch.nn.functional as F
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
    parser.add_argument('--pretrained', required=True, help='pretrained weight from checkpoint')
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
    testing(config, data_loader_test, model)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Test time {}'.format(total_time_str))


@torch.no_grad()
def testing(config, data_loader, model):
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()

    batch_time = AverageMeter()
    loss_meter = AverageMeter()

    end = time.time()
    evaluation = {}
    for idx, (images, target) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
            output = model(images)
        loss = criterion(output, target)

        _, predictions = torch.max(output, 1)
        outputs = torch.unbind(F.one_hot(predictions.cpu(), num_classes=5), dim=1)
        targets = torch.unbind(F.one_hot(target.cpu(), num_classes=5), dim=1)

        for out_id, (out, y) in enumerate(zip(outputs, targets)):
            pred, gt = out.numpy(), y.numpy()
            acc = accuracy_score(gt, pred) * 100
            f1 = f1_score(gt, pred, average="macro")
            precision = precision_score(gt, pred, average="macro")
            recall = recall_score(gt, pred, average="macro")

            indicator = evaluation.setdefault(f'class_{out_id}', {})
            indicator.setdefault('acc', AverageMeter()).update(acc, target.size(0))
            indicator.setdefault('f1', AverageMeter()).update(f1, target.size(0))
            indicator.setdefault('precision', AverageMeter()).update(precision, target.size(0))
            indicator.setdefault('recall', AverageMeter()).update(recall, target.size(0))

        loss_meter.update(loss.item(), target.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            message = ""
            for metric in evaluation['class_0'].keys():
                scores = [evaluation[cls][metric].val for cls in evaluation.keys()]
                score = sum(scores) / len(scores)
                avg_scores = [evaluation[cls][metric].avg for cls in evaluation.keys()]
                avg_score = sum(avg_scores) / len(avg_scores)
                message += f'{metric.upper()} {score:.4f} ({avg_score:.3f})\t'
            logger.info(
                f'Test: [{idx}/{len(data_loader)}]\t'
                f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                f'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t' + message + f'Mem {memory_used:.0f}MB')

    logger.info("Final test results:")

    message = f'Loss: {loss_meter.avg:.4f}\t'
    for metric in evaluation['class_0'].keys():
        scores = [evaluation[cls][metric].avg for cls in evaluation.keys()]
        avg_score = sum(scores) / len(scores)
        message += f'{metric.upper()} {avg_score:.3f}\t'
    logger.info(message)

    logger.info("Per class results:")
    for class_id in evaluation.keys():
        message = f'{class_id.upper()}: '
        for metric, indicator in evaluation[class_id].items():
            message += f'{metric.upper()} {indicator.avg:.4f}\t'
        logger.info(message)


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
