import pytest
from games.connect4 import Connect4
from util.solver import Solver
from util.supervised_mcts import SupervisedMCTS
import numpy as np
import torch


def test_supervised_mcts_search():
    model_path = "models/connect4_4x4_supervised.pt"
    iterations = 800
    supervised_mcts = SupervisedMCTS(model_path=model_path, iterations=iterations)

    game = Connect4(num_of_rows=4, num_of_cols=4)

    # search for the best move
    best_move = supervised_mcts.search(game)

    # root.num_visits should be same as iterations
    assert supervised_mcts.root.num_visits == iterations

    print("Best move found by Supervised MCTS:", best_move)

    assert best_move is not None

    # can we access the root?

    root = supervised_mcts.root
    assert root is not None

    print(root.game.board)

    # check n of children

    print("Number of children in root:", len(root.children))

    children = root.children
    assert len(children) > 0

    first = children[0]
    print("First child board state:")
    print(first.game.board)

    # legal moves
    legal_moves = game.get_legal_moves()
    for move in legal_moves:
        child = children[move]
        assert child is not None
        print("Child board state for move", move)
        child.game.print_pretty()
        print("Child visits:", child.num_visits)
        print("move_count:", child.game.move_count)
        assert child.turn == -1

    # assert root turn = 1
    assert root.turn == 1


def test_sup_mcts_search():
    board = np.array(
        [
            [0, 0, 0, 0],
            [1, -1, 0, 0],
            [1, -1, 0, 0],
            [1, -1, 0, 0],
        ],
        dtype=int,
    )
    game = Connect4(num_of_rows=4, num_of_cols=4, board=board)
    model_path = "models/connect4_4x4_supervised.pt"

    iterations = 800
    supervised_mcts = SupervisedMCTS(model_path=model_path, iterations=iterations)

    move_probs = supervised_mcts.search(game)

    print("Move probabilities:", move_probs)

    # argmax
    best_move = np.argmax(move_probs)
    print("Best move found by Supervised MCTS:", best_move)

    # assert best_move is not None
    # assert best_move == 0

    root = supervised_mcts.root

    policy_priors = supervised_mcts.evaluate_policy_with_model(root.game)
    node = root.PUCT(
        exploration_constant=supervised_mcts.exploration_constant,
        policy_priors=policy_priors,
        debug=True,
    )

    # best_child
    # best_child = root.best_child(debug=True)
    # assert best_child is not None
    # assert best_child.move == 123

    # legal_moves = game.get_legal_moves()
    # for move in legal_moves:
    #     print("================Legal move:", move)
    #     child = supervised_mcts.root.children[move]
    #     print("total score:", child.total_score)
    #     print("num visits:", child.num_visits)
    #     print("avg value:", child.total_score / child.num_visits)
    #     # assert child is not None
    # assert 1==2
