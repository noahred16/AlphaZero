from util.supervised_mcts import SupervisedMCTS, evaluate_supervised_mcts_on_test_data
from games.connect4 import Connect4
import torch
import numpy as np
from networks.Connect4Net import Connect4Net


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "models/connect4_4x4_alpha_zero_50k.pt"
# model_path = "models/connect4_4x4_supervised_100k.pt"
model = Connect4Net().to(device)
# model.load_state_dict(torch.load(model_path, map_location=device))
model.reset_weights()


iterations = 400
# iterations = 20
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
    # [
    #     [0, 1, 0, 0],
    #     [-1, -1, 0, -1],
    #     [1, -1, 0, 1],
    #     [-1, -1, 1, 1],
    # ],
    [[0, 0, 1, 0], [-1, 0, 1, -1], [1, 0, -1, -1], [-1, 1, 1, -1]]
]


# | | |X| |
# |O| |X|O|
# |X| |O|O|
# |O|X|X|O|
# ---------------
#  0 1 2 3

# Predicted move: 1 with value: 0.34375, but best moves were: [3]
# Policy: [0.32875 0.34375 0.      0.3275 ]

# for i, case in enumerate(cases):
#     board = case

#     game = Connect4(num_of_rows=4, num_of_cols=4, board=board)

#     supervised_mcts = SupervisedMCTS(
#         model=model,
#         iterations=iterations,
#         exploration_constant=exploration_constant,
#         selection_method="PUCT",
#     )

#     # priors
#     policy_priors = supervised_mcts.evaluate_policy_with_model(game)
#     # round to 3 decimal places
#     print("Policy priors:", np.round(policy_priors, 2))

#     # can test specific games
#     game.print_pretty()
#     move_probs = supervised_mcts.search(game)
#     print(f"Case {i+1}: Move probabilities: {move_probs}")

#     # creates a tree visualization png
#     supervised_mcts.tree_visualization()


# evaluate on test data
accuracy = evaluate_supervised_mcts_on_test_data(
    num_samples=1_000,
    mcts_iterations=iterations,
    exploration_constant=exploration_constant,
    selection_method="PUCT",
    # method="AlphaZero",
    model=model,
)
