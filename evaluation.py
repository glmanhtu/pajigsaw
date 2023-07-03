import argparse
import datetime
import glob
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
from torch.utils.data import Dataset

from config import get_config
from data.build import build_test_loader
from data.datasets.pieces_dataset import PiecesDataset
from data.transforms import TwoImgSyncEval
from logger import create_logger
from models import build_model
from paikin_tal_solver.puzzle_importer import Puzzle
from paikin_tal_solver.puzzle_piece import PuzzlePiece, PuzzlePieceSide
from solver_driver import paikin_tal_driver
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
    criterion = torch.nn.BCEWithLogitsLoss()
    model.eval()

    batch_time = AverageMeter()
    loss_meter = AverageMeter()

    end = time.time()
    evaluation = {}

    for subset in ['BGU', 'Cho', 'McGill']:
        images = glob.glob(os.path.join(config.DATA.DATA_PATH, subset, '*.jpg'))
        images += glob.glob(os.path.join(config.DATA.DATA_PATH, subset, '*.png'))

        perfect_predictions, direct_accuracies, neighbour_accuracies = [], [], []
        for img_path in images:
            puzzle = Puzzle(0, img_path, config.DATA.IMG_SIZE, starting_piece_id=0)
            pieces = puzzle.pieces
            random.shuffle(pieces)
            dataset = PiecesDataset(pieces, transform=TwoImgSyncEval(config.DATA.IMG_SIZE),
                                    image_size=config.DATA.IMG_SIZE, erosion_ratio=config.DATA.EROSION_RATIO)
            sampler_val = torch.utils.data.distributed.DistributedSampler(
                dataset, shuffle=config.TEST.SHUFFLE
            )
            data_loader = torch.utils.data.DataLoader(
                dataset, sampler=sampler_val,
                batch_size=config.DATA.BATCH_SIZE,
                shuffle=False,
                num_workers=config.DATA.NUM_WORKERS,
                pin_memory=config.DATA.PIN_MEMORY,
                drop_last=False
            )

            distance_map = {}
            for idx, (images, target) in enumerate(data_loader):
                images = images.cuda(non_blocking=True)

                # compute output
                with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
                    output = model(images)

                for pred, entry_id in zip(torch.sigmoid(output), target):
                    i, j = dataset.entries[entry_id]
                    piece_i, piece_j = pieces[i].origin_piece_id, pieces[j].origin_piece_id
                    if piece_i not in distance_map:
                        distance_map[piece_i] = {}
                    distance_map[piece_i][piece_j] = 1. - pred

            def distance_function(piece_i, piece_i_side, piece_j, piece_j_side):
                pred = distance_map[piece_i.origin_piece_id][piece_j.origin_piece_id]
                if piece_j_side == PuzzlePieceSide.left:
                    if piece_i_side == PuzzlePieceSide.right:
                        return pred[1].item()
                if piece_j_side == PuzzlePieceSide.right:
                    if piece_i_side == PuzzlePieceSide.left:
                        return pred[3].item()
                if piece_j_side == PuzzlePieceSide.top:
                    if piece_i_side == PuzzlePieceSide.bottom:
                        return pred[0].item()
                if piece_j_side == PuzzlePieceSide.bottom:
                    if piece_i_side == PuzzlePieceSide.top:
                        return pred[2].item()
                return 1.

            perfect_pred, direct_acc, neighbour_acc, new_puzzle = paikin_tal_driver(pieces, distance_function)
            perfect_predictions.append(perfect_pred)
            direct_accuracies.append(direct_acc)
            neighbour_accuracies.append(neighbour_acc)

            output_dir = os.path.join('output', 'reconstructed')
            os.makedirs(output_dir, exist_ok=True)
            new_puzzle.save_to_file(os.path.join(output_dir, os.path.basename(img_path)))

        print(f'Total perfect_acc: {sum(perfect_predictions)} / {len(perfect_predictions)}')
        print(f'Avg direct_acc: {sum(direct_accuracies) / len(direct_accuracies)}')
        print(f'Avg neighbour_acc: {sum(neighbour_accuracies) / len(neighbour_accuracies)}')


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
