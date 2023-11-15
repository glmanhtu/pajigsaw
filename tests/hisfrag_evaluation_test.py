import argparse
import datetime
import os
import time
import torch.nn.functional as F
import numpy as np
import torch
import torchvision
from timm.utils import AverageMeter
from torch.utils.data import Dataset

from data.datasets.hisfrag_dataset import HisFrag20GT
from data.samplers import DistributedEvalSampler
from hisfrag import HisfragTrainer
from misc import wi19_evaluate, utils


@torch.no_grad()
def eval_standard(config, model, logger, world_size, rank):
    model.eval()
    transform = torchvision.transforms.Compose([
        torchvision.transforms.CenterCrop(512),
        torchvision.transforms.Resize(config.DATA.IMG_SIZE),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    dataset = HisFrag20GT(config.DATA.DATA_PATH, HisFrag20GT.Split.VAL, transform=transform,
                          val_n_items_per_writer=config.DATA.EVAL_N_ITEMS_PER_CATEGORY)
    sampler = DistributedEvalSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = torch.utils.data.DataLoader(
        dataset, sampler=sampler,
        batch_size=config.DATA.TEST_BATCH_SIZE,
        shuffle=False,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=True,
        drop_last=False
    )

    predicts = torch.zeros((0, 3), dtype=torch.float32).cuda()

    batch_time = AverageMeter()
    end = time.time()
    for idx, (images, pair) in enumerate(dataloader):
        images = images.cuda(non_blocking=True)
        with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
            output = model(images)

        sub_pair_predicts = torch.column_stack([pair.cuda().float(), output.float()])
        predicts = torch.cat([predicts, sub_pair_predicts])
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

    if world_size > 1:
        max_n_items = int(len(dataset.pairs) * 1.2 / world_size)
        # create an empty list we will use to hold the gathered values
        predicts_list = [torch.zeros((max_n_items, 3), dtype=torch.float32, device=predicts.device)
                         for _ in range(world_size)]
        # Tensors from different processes have to have the same N items, therefore we pad it with -1
        predicts = F.pad(predicts, pad=(0, 0, 0, max_n_items - predicts.shape[0]), mode="constant", value=-1)

        # sending all tensors to the others
        torch.distributed.barrier()
        torch.distributed.all_gather(predicts_list, predicts, async_op=False)

        # Remove all padded items
        predicts_list = [x[x[:, 0] != -1] for x in predicts_list]
        predicts = torch.cat(predicts_list, dim=0)

    predicts = predicts.cpu()

    assert len(predicts) == len(dataset.pairs), f'Incorrect size {predicts.shape} vs {dataset.pairs.shape}'
    size = len(dataset.samples)

    # Initialize a similarity matrix with zeros
    similarity_matrix = torch.zeros((size, size), dtype=torch.float16)

    # Extract index pairs and scores
    indices = predicts[:, :2].long()
    scores = predicts[:, 2].type(torch.float16)

    # Use indexing and broadcasting to fill the similarity matrix
    similarity_matrix[indices[:, 0], indices[:, 1]] = scores
    similarity_matrix[indices[:, 1], indices[:, 0]] = scores

    # max_score = torch.amax(similarity_matrix, dim=1)
    distance_matrix = 1 - similarity_matrix

    labels = []
    for i in range(size):
        labels.append(os.path.splitext(os.path.basename(dataset.samples[i]))[0])
    return distance_matrix.numpy(), labels


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Pajigsaw testing script', add_help=False)
    parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    # easy config modification
    parser.add_argument('--mode', type=str, default='eval')
    parser.add_argument('--batch-size', type=int, help="batch size for data")
    parser.add_argument('--eval-n-items-per-category', type=int, default=5,
                        help="Number of items per category to test")
    parser.add_argument('--data-path', type=str, help='path to dataset')
    parser.add_argument('--disable_amp', action='store_true', help='Disable pytorch amp')
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--tag', help='tag of experiment')

    args, unparsed = parser.parse_known_args()

    trainer = HisfragTrainer(args)
    logger = trainer.logger

    logger.info("Start testing")
    m_ap = 1 - trainer.validate()

    start_time = time.time()
    distance_matrix, img_names = eval_standard(trainer.config, trainer.model, trainer.logger,
                                               trainer.world_size, trainer.rank)
    labels = utils.list_to_idx(img_names, lambda x: x.split('_')[0])
    logger.info('Starting to calculate performance...')
    m_ap2, top1, pr_a_k10, pr_a_k100 = wi19_evaluate.get_metrics(distance_matrix, np.asarray(labels))
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info(f'Original approach: mAP {m_ap2:.3f}\t' f'Top 1 {top1:.3f}\t' f'Pr@k10 {pr_a_k10:.3f}\t'
                f'Pr@k100 {pr_a_k100:.3f} Time: {total_time_str}')

    logger.info(f'First: {m_ap}, second: {m_ap2}')
    np.testing.assert_almost_equal(m_ap, m_ap2)
