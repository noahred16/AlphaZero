import pytest
from games.connect4 import Connect4
from util.solver import Solver
from util.mcts import MCTS
import numpy as np


def test_connect4_evaluate_board():
    # board = [
    #     [0, 0, 0, 0],
    #     [1, -1, 0, 0],
    #     [1, -1, 0, 0],
    #     [1, -1, 0, 0],
    # ]
    board = np.array(
        [
            [1, 0, 0, 0],
            [1, -1, 0, 0],
            [1, -1, 0, 0],
            [1, -1, 0, 0],
        ],
        dtype=int,
    )
    game = Connect4(num_of_rows=4, num_of_cols=4, board=board)
    assert game.move_count == 7

    board_value = game.evaluate_board()
    print("moves made: ", game.move_count)

    # full board (16) - num of moves (7) = 9
    assert board_value == 9

    opp_win = -board
    print("Opponent's winning board:")
    print(opp_win)
    game = Connect4(num_of_rows=4, num_of_cols=4, board=opp_win)
    board_value = game.evaluate_board(player=-1)
    assert game.move_count == 7
    assert board_value == 9


def test_tie_game():
    board = np.array(
        [
            [-1, 1, 1, -1],
            [-1, 1, -1, 1],
            [1, -1, 1, -1],
            [-1, 1, -1, 1],  # full board with no winner
        ],
        dtype=int,
    )
    game = Connect4(num_of_rows=4, num_of_cols=4, board=board)
    game.print_pretty()  # for visual confirmation
    board_value = game.evaluate_board()
    assert game.move_count == 16  # full board
    assert board_value == 0  # tie game


def test_play_game():
    game = Connect4(num_of_rows=4, num_of_cols=4)
    game.make_move(0)
    game.make_move(0)
    game.make_move(0)
    game.make_move(0)
    game.make_move(1)
    game.make_move(1)
    game.make_move(1)
    game.make_move(1)

    game.make_move(2)
    game.make_move(3)
    game.make_move(3)
    game.make_move(2)
    game.make_move(3)
    game.make_move(2)
    game.make_move(3)
    game.make_move(2)

    val = game.evaluate_board()

    assert val == 0  # tie game, no winner

    # game.print_pretty()  # for visual confirmation

    # assert game.move_count == 8  # 4 moves each player


def test_eval():
    # |O|O|O| |
    # |X|X|X| |
    # |O|O|X|O|
    # |X|X|X|O|
    # ---------------
    # 0 1 2 3

    board = np.array(
        [
            [-1, -1, 0, 0],  # O wins
            [1, 1, 1, -1],  # X wins
            [-1, -1, 1, -1],
            [1, 1, 1, -1],  # full board with no winner
        ],
        dtype=int,
    )
    game = Connect4(num_of_rows=4, num_of_cols=4, board=board)
    game.print_pretty()  # for visual confirmation
    board_value = game.evaluate_board(player=1)
    assert board_value == None  # tie game, no winner
    # assert game.move_count == 12  # full board


def test_solver_flipping():
    """
    we want to visualize the board the same regarless of whose turn it is.
    so instead of keeping track of turns, we just flip the board 1's to -1's after each move
    """
    game = Connect4(num_of_rows=4, num_of_cols=4)
    # X = 1, O = -1
    # [ _ | _ | _ | _ ]
    # [ _ | 0 | _ | _ ]
    # [ X | O | _ | _ ]
    # [ X | O | X | _ ]
    #   0   1   2   3

    # value: 1 (X wins next move)
    # policy: [0.68, 0.32, 0, 0]

    game.make_move(0)
    # game.flip_board()  # flip the board to visualize the next player's turn
    assert game.board[3][0] == -1

    game.make_move(1)
    # game.flip_board()  # flip the board to visualize the next player's turn
    assert game.board[3][0] == 1
    assert game.board[3][1] == -1

def test_connect4_undo():
    game = Connect4(num_of_rows=4, num_of_cols=4)
    game.make_move(0)
    game.make_move(1)

    assert game.board[3][0] == 1
    assert game.board[3][1] == -1
    assert game.move_count == 2

    game.undo_move()

    assert game.board[3][0] == -1
    assert game.board[3][1] == 0
    assert game.move_count == 1
