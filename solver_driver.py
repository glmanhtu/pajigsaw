"""Main Puzzle Solver Driver
   Adapted from https://github.com/ZaydH/sjsu_thesis
.. moduleauthor:: Zayd Hammoudeh <hammoudeh@gmail.com>
"""
import os.path
import random

from paikin_tal_solver.puzzle_importer import Puzzle, PuzzleType
from paikin_tal_solver.puzzle_piece import PuzzlePiece
from paikin_tal_solver.solver import PaikinTalSolver


def paikin_tal_driver(image_file, puzzle_type=PuzzleType.type1, piece_width=28, show_image=True):
    """
    Runs the Paikin and Tal image solver.

    Args:
        image_file ([str]): path to image
        puzzle_type (Optional PuzzleType): Type of the puzzle to solve
        piece_width (Optional int): Width of a puzzle piece in pixels.
        show_image: visualize the solved image
    """

    new_puzzle = Puzzle(0, image_file, piece_width, starting_piece_id=0)

    pieces = new_puzzle.pieces
    # For good measure, shuffle the pieces
    random.shuffle(pieces)

    # Create the Paikin Tal Solver
    paikin_tal_solver = PaikinTalSolver(1, pieces, PuzzlePiece.calculate_asymmetric_distance, puzzle_type)

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
    new_puzzle = Puzzle.reconstruct_from_pieces(puzzle_pieces, puzzle_id)

    # Optionally display the images
    if show_image:
        # noinspection PyProtectedMember
        Puzzle.display_image(new_puzzle._img)

    # Store the reconstructed image
    output_dir = os.path.join('output', "reconstructed_type_" + str(paikin_tal_solver.puzzle_type.value))
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, os.path.basename(image_file))

    new_puzzle.save_to_file(filename)


if __name__ == "__main__":
    image = os.path.join(os.path.dirname(__file__), 'images', '7.jpg')
    paikin_tal_driver(image, PuzzleType.type1, 64)
