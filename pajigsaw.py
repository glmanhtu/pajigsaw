import argparse
import os
import random
import time

import torch
from torch.utils.data import DataLoader

from data.datasets.pajigsaw_dataset import PajigsawPieces, Pajigsaw
from data.datasets.pieces_dataset import PiecesDataset
from data.transforms import TwoImgSyncEval
from misc.engine import Trainer
from misc.utils import AverageMeter
from paikin_tal_solver.puzzle_importer import PuzzleResultsCollection, PuzzleSolver, PuzzleType
from paikin_tal_solver.puzzle_piece import PuzzlePieceSide
from solver_driver import paikin_tal_driver


def parse_option():
    parser = argparse.ArgumentParser('Pajigsaw training and evaluation script', add_help=False)
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
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--disable_amp', action='store_true', help='Disable pytorch amp')
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--mode', type=str, choices=['train', 'eval', 'test', 'throughput'], default='train')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')

    # overwrite optimizer in config (*.yaml) if specified, e.g., fused_adam/fused_lamb
    parser.add_argument('--optim', type=str,
                        help='overwrite optimizer if provided, can be adamw/sgd/fused_adam/fused_lamb.')

    return parser.parse_known_args()


class PajigsawTrainer(Trainer):

    def get_criterion(self):
        return torch.nn.BCEWithLogitsLoss()

    def validate_dataloader(self, dataset):
        self.model.eval()
        puzzles = []
        im_names = []
        batch_time = AverageMeter()
        end = time.time()
        for idx, (pieces, im_name, grid_size) in enumerate(dataset):
            random.shuffle(pieces)
            im_names.append(im_name)
            pieces_dataset = PiecesDataset(pieces, transform=TwoImgSyncEval(self.config.DATA.IMG_SIZE))
            data_loader = DataLoader(
                pieces_dataset,
                batch_size=self.config.DATA.BATCH_SIZE,
                shuffle=False,
                num_workers=self.config.DATA.NUM_WORKERS,
                pin_memory=self.config.DATA.PIN_MEMORY,
                drop_last=False
            )

            distance_map = {}
            for images, targets in data_loader:
                images = images.cuda(non_blocking=True)

                # compute output
                with torch.cuda.amp.autocast(enabled=self.config.AMP_ENABLE):
                    output = self.model(images)

                for pred, entry_id in zip(torch.sigmoid(output).cpu().numpy(), targets.numpy()):
                    i, j = pieces_dataset.entries[entry_id]
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

            new_puzzle = paikin_tal_driver(pieces, self.config.DATA.IMG_SIZE, distance_function, grid_size)
            puzzles.append(new_puzzle)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if idx % self.config.PRINT_FREQ == 0:
                memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
                self.logger.info(
                    f'Eval: [{idx}/{len(dataset)}]\t'
                    f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    f'Mem {memory_used:.0f}MB')

        results_information = PuzzleResultsCollection(PuzzleSolver.PaikinTal, PuzzleType.type1,
                                                      [x.pieces for x in puzzles], im_names)

        # Calculate and print the accuracy results
        results_information.calculate_accuracies(puzzles)
        # Print the results to the console
        result, perfect_puzzles = results_information.collect_results()

        out = 'Average_Results:\t'
        for key in result:
            out += f'{key}: {round(sum(result[key]) / len(result[key]), 4)}\t'
        out += f'Perfect: {sum(perfect_puzzles)}'
        self.logger.info(out)
        return sum(result['neighbor']) / len(result['neighbor']), puzzles, im_names

    @torch.no_grad()
    def test(self):
        self.logger.info("Starting test...")
        dataset = PajigsawPieces(self.config.DATA.DATA_PATH, Pajigsaw.Split.TEST)
        _, puzzles, im_names = self.validate_dataloader(dataset)

        for puzzle, im_name in zip(puzzles, im_names):
            output_file = os.path.join(self.config.OUTPUT, 'reconstructed', f'{im_name}.jpg')
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            puzzle.save_to_file(output_file)

    @torch.no_grad()
    def validate(self):
        self.logger.info("Starting validation...")
        dataset = PajigsawPieces(self.config.DATA.DATA_PATH, Pajigsaw.Split.VAL)
        neighbor_precision, _, _ = self.validate_dataloader(dataset)
        return 1 - neighbor_precision


if __name__ == '__main__':
    args, _ = parse_option()
    trainer = PajigsawTrainer(args)
    if args.mode == 'eval':
        trainer.validate()
    elif args.mode == 'test':
        trainer.test()
    elif args == 'throughput':
        trainer.throughput()
    else:
        trainer.train()
