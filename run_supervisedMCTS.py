from util.supervised_mcts import SupervisedMCTS, evaluate_supervised_mcts_on_test_data
from games.connect4 import Connect4
import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "models/connect4_4x4_supervised_100k.pt"

iterations = 800
exploration_constant = 1.5

cases = [
    # # should select ind 0
    # [
    #     [0, 0, 0, 0],
    #     [1, -1, 0, 0],
    #     [1, -1, 0, 0],
    #     [1, -1, 0, 0],
    # ],
    # # correct move is ind 3
    # [
    #     [0, 0, 1, 0],
    #     [0, 1, -1, -1],
    #     [1, -1, 1, 1],
    #     [-1, -1, -1, 1],
    # ],
    # [
    #     [-1, 0,  0,  -1],
    #     [ 1, 0,  0,   1],
    #     [-1, 1, -1,   1],
    #     [-1, 1, -1,   1],
    # ],
    [
        [0, 1, 0, 0],
        [-1, -1, 0, -1],
        [1, -1, 0, 1],
        [-1, -1, 1, 1],
    ]
]


for i, case in enumerate(cases):
    board = case

    game = Connect4(num_of_rows=4, num_of_cols=4, board=board)

    supervised_mcts = SupervisedMCTS(
        model_path=model_path,
        device=device,
        iterations=iterations,
        exploration_constant=exploration_constant,
    )

    # can test specific games
    game.print_pretty()
    move_probs = supervised_mcts.search(game)
    print(f"Case {i+1}: Move probabilities: {move_probs}")

    # creates a tree visualization png
    supervised_mcts.tree_visualization()


# evaluate on test data
accuracy = evaluate_supervised_mcts_on_test_data(
    num_samples=20_000,
    mcts_iterations=iterations,
    exploration_constant=exploration_constant,
)
