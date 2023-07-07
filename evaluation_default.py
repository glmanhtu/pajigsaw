import argparse
import glob
import os
import random

from paikin_tal_solver.puzzle_importer import Puzzle
from paikin_tal_solver.puzzle_piece import PuzzlePiece
from solver_driver import paikin_tal_driver


def parse_option():
    parser = argparse.ArgumentParser('Pajigsaw testing script', add_help=False)
    parser.add_argument('--data-path', type=str, help='path to dataset', required=True)
    parser.add_argument('--image-size', type=int, default=64)
    parser.add_argument('--erosion', type=float, default=0.07)

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_option()

    for subset in ['Cho', 'McGill', 'BGU']:
        images = glob.glob(os.path.join(args.data_path, subset, '*.jpg'))
        images += glob.glob(os.path.join(args.data_path, subset, '*.png'))

        perfect_predictions, direct_accuracies, neighbour_accuracies = [], [], []
        for img_path in images:
            puzzle = Puzzle(0, img_path, args.image_size, starting_piece_id=0, erosion=args.erosion)
            pieces = puzzle.pieces
            random.shuffle(pieces)

            def distance_function(piece_i, piece_i_side, piece_j, piece_j_side):
                return PuzzlePiece.calculate_asymmetric_distance(piece_i, piece_i_side, piece_j, piece_j_side)

            perfect_pred, direct_acc, neighbour_acc, new_puzzle = paikin_tal_driver(pieces, args.image_size,
                                                                                    distance_function)
            perfect_predictions.append(perfect_pred)
            direct_accuracies.append(direct_acc)
            neighbour_accuracies.append(neighbour_acc)

            output_dir = os.path.join('output', 'reconstructed')
            os.makedirs(output_dir, exist_ok=True)
            new_puzzle.save_to_file(os.path.join(output_dir, os.path.basename(img_path)))

        print(f'Total perfect_acc: {sum(perfect_predictions)} / {len(perfect_predictions)}')
        print(f'Avg direct_acc: {sum(direct_accuracies) / len(direct_accuracies)}')
        print(f'Avg neighbour_acc: {sum(neighbour_accuracies) / len(neighbour_accuracies)}')
