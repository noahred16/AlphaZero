import pytest
from games.connect4 import Connect4
from util.solver import Solver


def test_solver_evaluate_state():
    game = Connect4(num_of_rows=4, num_of_cols=4)
    solver = Solver(game)

    # X = 1, O = -1
    # [ _ | _ | _ | _ ]
    # [ X | 0 | _ | _ ]
    # [ X | O | _ | _ ]
    # [ X | O | _ | _ ]
    #   0   1   2   3

    # value: 1 (X wins next move)
    # policy: [0.68, 0.32, 0, 0]

    game.make_move(0)
    game.make_move(1)
    game.make_move(0)
    game.make_move(1)
    game.make_move(0)
    game.make_move(1)

    game.print_pretty()

    policy_label, value_label = solver.evaluate_state()

    assert value_label == 1  # scaled between best and worst score
    assert round(policy_label[0], 2) == 0.68  # best move
    assert round(policy_label[1], 2) == 0.32  # okay move, leads to tie
    assert round(policy_label[2], 2) == 0.0  # bad move, leads to loss
    assert round(policy_label[3], 2) == 0.0  # bad move, leads to loss
    assert round(sum(policy_label), 2) == 1.0  # assert probabilities sum to 1


def test_solver_empty_board():
    game = Connect4(num_of_rows=4, num_of_cols=4)
    solver = Solver(game)

    # evaluation for empty board
    policy_label, value_label = solver.evaluate_state()

    assert value_label == 0  # all moves lead to a tie given optimally play
    # assert all moves are 0.25 rounded
    for i in range(4):
        assert round(policy_label[i], 2) == 0.25
