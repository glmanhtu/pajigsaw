from paikin_tal_solver.puzzle_importer import Puzzle
from paikin_tal_solver.puzzle_piece import PuzzlePieceSide


def compute_neighbor_accuracy(puzzle: Puzzle):
    placement_matrix = puzzle.build_placed_piece_info()
    cols, rows = placement_matrix.shape
    correct_neighbor = 0
    total_neighbor = 0
    for i in range(cols):
        for j in range(rows):
            if i == j :
                continue
            piece = puzzle.pieces[placement_matrix[i][j]]
            if j > 0:
                left_piece = puzzle.pieces[placement_matrix[i][j - 1]]
                correct_neighbor += piece.is_neighbor(left_piece, PuzzlePieceSide.left)
                total_neighbor += 1
            if j < rows - 1:
                right_piece = puzzle.pieces[placement_matrix[i][j + 1]]
                correct_neighbor += piece.is_neighbor(right_piece, PuzzlePieceSide.right)
                total_neighbor += 1

            if i > 0:
                top_piece = puzzle.pieces[placement_matrix[i - 1][j]]
                correct_neighbor += piece.is_neighbor(top_piece, PuzzlePieceSide.top)
                total_neighbor += 1

            if i < cols - 1:
                bottom_piece = puzzle.pieces[placement_matrix[i + 1][j]]
                correct_neighbor += piece.is_neighbor(bottom_piece, PuzzlePieceSide.bottom)
                total_neighbor += 1
    return correct_neighbor / total_neighbor


def compute_direct_accuracy(puzzle: Puzzle):
    correct_placement = 0
    total_placement = 0
    for piece in puzzle.pieces:
        if piece.is_correct_placement():
            correct_placement += 1
        total_placement += 1
    return correct_placement / total_placement
