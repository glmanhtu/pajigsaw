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
from torch.utils.data import Dataset

from config import get_config
from data.datasets.pieces_dataset import PiecesDataset
from data.transforms import TwoImgSyncEval
from misc.logger import create_logger
from models import build_model
from paikin_tal_solver.puzzle_importer import Puzzle, PuzzleResultsCollection, PuzzleSolver, PuzzleType
from paikin_tal_solver.puzzle_piece import PuzzlePieceSide
from solver_driver import paikin_tal_driver
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
    capability = torch.cuda.get_device_capability()
    if capability[0] >= 7:
        model = torch.compile(model)

    for subset in ['Cho', 'McGill', 'BGU']:
        images = glob.glob(os.path.join(config.DATA.DATA_PATH, subset, '*.jpg'))
        images += glob.glob(os.path.join(config.DATA.DATA_PATH, subset, '*.png'))

        puzzles = []
        for idx, img_path in enumerate(images):
            puzzle = Puzzle(idx, img_path, config.DATA.IMG_SIZE, starting_piece_id=0, erosion=config.DATA.EROSION_RATIO)
            pieces = puzzle.pieces
            random.shuffle(pieces)
            dataset = PiecesDataset(pieces, transform=TwoImgSyncEval(config.DATA.IMG_SIZE))
            data_loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=config.DATA.BATCH_SIZE,
                shuffle=False,
                num_workers=config.DATA.NUM_WORKERS,
                pin_memory=config.DATA.PIN_MEMORY,
                drop_last=False
            )

            distance_map = {}
            for images, targets in data_loader:
                images = images.cuda(non_blocking=True)

                # compute output
                with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
                    output = model(images)

                for pred, entry_id in zip(torch.sigmoid(output).cpu().numpy(), targets.numpy()):
                    i, j = dataset.entries[entry_id]
                    piece_i, piece_j = pieces[i].origin_piece_id, pieces[j].origin_piece_id
                    if piece_i not in distance_map:
                        distance_map[piece_i] = {}
                    distance_map[piece_i][piece_j] = 1. - pred

            def distance_function(piece_i, piece_i_side, piece_j, piece_j_side):
                nonlocal distance_map
                pred = distance_map[piece_i.origin_piece_id][piece_j.origin_piece_id]
                if piece_j_side == PuzzlePieceSide.left:
                    if piece_i_side == PuzzlePieceSide.right:
                        return pred[0] * 1000.
                if piece_j_side == PuzzlePieceSide.right:
                    if piece_i_side == PuzzlePieceSide.left:
                        return pred[2] * 1000.
                if piece_j_side == PuzzlePieceSide.top:
                    if piece_i_side == PuzzlePieceSide.bottom:
                        return pred[1] * 1000.
                if piece_j_side == PuzzlePieceSide.bottom:
                    if piece_i_side == PuzzlePieceSide.top:
                        return pred[3] * 1000.
                return float('inf')

            new_puzzle = paikin_tal_driver(pieces, config.DATA.IMG_SIZE, distance_function, puzzle.grid_size)
            puzzles.append(new_puzzle)

            output_dir = os.path.join('output', 'reconstructed', subset)
            os.makedirs(output_dir, exist_ok=True)
            new_puzzle.save_to_file(os.path.join(output_dir, os.path.basename(img_path)))

        print(f'Subset: {subset} {len(puzzles[0].pieces)}')
        results_information = PuzzleResultsCollection(PuzzleSolver.PaikinTal, PuzzleType.type1,
                                                      [x.pieces for x in puzzles], images)

        # Calculate and print the accuracy results
        results_information.calculate_accuracies(puzzles)
        # Print the results to the console
        result, perfect_puzzles = results_information.collect_results()

        out = 'Average_Results:\t'
        for key in result:
            out += f'{key}: {round(sum(result[key]) / len(result[key]), 4)}\t'
        out += f'Perfect: {sum(perfect_puzzles)}'
        logger.info(out)


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
