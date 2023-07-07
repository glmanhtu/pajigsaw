"""Main Puzzle Solver Driver
   Adapted from https://github.com/ZaydH/sjsu_thesis
.. moduleauthor:: Zayd Hammoudeh <hammoudeh@gmail.com>
"""
import glob
import os.path
import random

from paikin_tal_solver import puzzle_evaluation
from paikin_tal_solver.puzzle_importer import Puzzle, PuzzleType, PuzzleResultsCollection, PuzzleSolver
from paikin_tal_solver.puzzle_piece import PuzzlePiece
from paikin_tal_solver.solver import PaikinTalSolver


def paikin_tal_driver(pieces, piece_width, distance_fn):

    # Create the Paikin Tal Solver
    paikin_tal_solver = PaikinTalSolver(1, pieces, distance_fn, PuzzleType.type1)

    # Run the Solver
    paikin_tal_solver.run()

    # Get the results
    (paikin_tal_results, _) = paikin_tal_solver.get_solved_puzzles()

    # Print the Paikin Tal Solver Results
    puzzle_pieces = paikin_tal_results[0]
    # Get the first piece of the puzzle and extract information on it.
    first_piece = puzzle_pieces[0]
    puzzle_id = first_piece.puzzle_id

    # Reconstruct the puzzle
    return Puzzle.reconstruct_from_pieces(puzzle_pieces, piece_width, puzzle_id)


if __name__ == "__main__":
    images = glob.glob(os.path.join(os.path.dirname(__file__), 'images', "*.jpg"))
    perfect_predictions, direct_accuracies, neighbour_accuracies = [], [], []
    piece_width = 64

    for img_path in images:
        puzzle = Puzzle(0, img_path, piece_width, starting_piece_id=0, erosion=0.07)
        pieces = puzzle.pieces
        random.shuffle(pieces)

        def distance_function(piece_i, piece_i_side, piece_j, piece_j_side):
            return PuzzlePiece.calculate_asymmetric_distance(piece_i, piece_i_side, piece_j, piece_j_side)


        new_puzzle = paikin_tal_driver(pieces, piece_width, distance_function)
        results_information = PuzzleResultsCollection(PuzzleSolver.PaikinTal, PuzzleType.type1,
                                                      [new_puzzle.pieces], [img_path])
        # Calculate and print the accuracy results
        results_information.calculate_accuracies([new_puzzle])
        # Print the results to the console
        results_information.print_results()

        output_dir = os.path.join('output', 'reconstructed')
        os.makedirs(output_dir, exist_ok=True)
        new_puzzle.save_to_file(os.path.join(output_dir, os.path.basename(img_path)))

    print(f'Total perfect_acc: {sum(perfect_predictions)} / {len(perfect_predictions)}')
    print(f'Avg direct_acc: {sum(direct_accuracies) / len(direct_accuracies)}')
    print(f'Avg neighbour_acc: {sum(neighbour_accuracies) / len(neighbour_accuracies)}')
