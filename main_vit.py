import argparse
import datetime
import glob
import os
import random
import time

import torch
import torch.nn.functional as F
import torchvision

from data.datasets.pieces_dataset import PiecesDataset, PiecesDatasetTriplet
from data.transforms import TwoImgSyncEval
from misc.engine import Trainer
from misc.utils import AverageMeter
from paikin_tal_solver.puzzle_importer import PuzzleResultsCollection, PuzzleSolver, PuzzleType, Puzzle
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
    parser.add_argument('--puzzle-data-path', type=str, help='path to dataset')
    parser.add_argument('--pretrained',
                        help='pretrained weight from checkpoint, could be imagenet22k pretrained weight')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--disable_amp', action='store_true', help='Disable pytorch amp')
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--mode', type=str, choices=['train', 'eval', 'throughput', 'test'], default='train')

    # overwrite optimizer in config (*.yaml) if specified, e.g., fused_adam/fused_lamb
    parser.add_argument('--optim', type=str,
                        help='overwrite optimizer if provided, can be adamw/sgd/fused_adam/fused_lamb.')

    return parser.parse_known_args()


def distance_fn(x, y):
    return 1.0 - F.cosine_similarity(x, y, dim=-1)


class TripletLoss(torch.nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.loss_fn = torch.nn.TripletMarginWithDistanceLoss(
            distance_function=distance_fn, margin=margin)

    def forward(self, features, target):
        return self.loss_fn(features[:, 0, :], features[:, 1, :], features[:, 2, :])


class DefaultTrainer(Trainer):

    def get_transforms(self):
        patch_size = self.config.DATA.IMG_SIZE
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(patch_size),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        return {
            'train': transform,
            'validation': transform,
            'test': transform
        }

    def train_step(self, samples):
        B, X, S, C, H, W = samples.shape
        output = self.model(samples.view((B * S * X, C, H, W)))
        return output.view((B * X, S, -1))

    def get_criterion(self):
        return TripletLoss(margin=0.2)

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        data_loader = self.get_dataloader('validation')
        criterion = self.get_criterion()
        batch_time = AverageMeter()
        loss_meter = AverageMeter()

        start = time.time()
        end = time.time()
        for idx, (images, targets) in enumerate(data_loader):
            images = images.cuda(non_blocking=True)
            # target = target.cuda(non_blocking=True)

            # compute output
            with torch.cuda.amp.autocast(enabled=self.config.AMP_ENABLE):
                output = self.train_step(images)

            loss = criterion(output, targets)

            loss_meter.update(loss.item(), images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % self.config.PRINT_FREQ == 0:
                memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
                self.logger.info(
                    f'Eval: [{idx}/{len(data_loader)}]\t'
                    f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    f'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                    f'Mem {memory_used:.0f}MB')

        # Gathering results from gpus
        torch.distributed.barrier()
        loss_meter.all_reduce()
        batch_time.all_reduce()
        test_time = datetime.timedelta(seconds=int(time.time() - start))

        self.logger.info(
            f'Overall:'
            f'Time {test_time}\t'
            f'Batch Time {batch_time.avg:.3f}\t'
            f'Loss {loss_meter.avg:.4f}\t')

        return loss_meter.avg

    @torch.no_grad()
    def testing(self):
        model = self.model
        config = self.config
        model.eval()
        # capability = torch.cuda.get_device_capability()
        # if capability[0] >= 7:
        #     model = torch.compile(model)

        for subset in ['Cho', 'McGill', 'BGU']:
            images = glob.glob(os.path.join(config.DATA.DATA_PATH, subset, '*.jpg'))
            images += glob.glob(os.path.join(config.DATA.DATA_PATH, subset, '*.png'))

            puzzles = []
            for idx, img_path in enumerate(images):
                puzzle = Puzzle(idx, img_path, config.DATA.IMG_SIZE, starting_piece_id=0,
                                erosion=config.DATA.EROSION_RATIO)
                pieces = puzzle.pieces
                random.shuffle(pieces)
                dataset = PiecesDatasetTriplet(pieces, transform=TwoImgSyncEval(config.DATA.IMG_SIZE))
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
                        B, S, C, H, W = images.shape
                        output = model(images.view((B * S, C, H, W)))
                        output = output.view((B, S // 2, 2, -1))
                        first_features = output[:, :, 0, :]
                        second_features = output[:, :, 1, :]
                        distances = distance_fn(first_features, second_features)

                    for pred, entry_id in zip(distances.cpu().numpy(), targets.numpy()):
                        i, j = dataset.entries[entry_id]
                        piece_i, piece_j = pieces[i].origin_piece_id, pieces[j].origin_piece_id
                        if piece_i not in distance_map:
                            distance_map[piece_i] = {}
                        distance_map[piece_i][piece_j] = pred

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
            self.logger.info(out)


if __name__ == '__main__':
    args, _ = parse_option()
    trainer = DefaultTrainer(args)
    if args.mode == 'eval':
        trainer.validate()
    elif args.mode == 'throughput':
        trainer.throughput()
    elif args.mode == 'test':
        trainer.testing()
    else:
        trainer.train()
