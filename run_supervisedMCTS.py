from util.supervised_mcts import SupervisedMCTS
from games.connect4 import Connect4
import random
import math
import numpy as np
import copy
import torch
import torch.nn as nn


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mcts = SupervisedMCTS(model_path="models/connect4_4x4_supervised.pt", iterations=500, device=device)
game = Connect4(num_of_rows=4, num_of_cols=4)
move_probs = mcts.search(game)
print("Move probabilities:", move_probs)
samples = mcts.get_samples()


# evaluate_supervised_mcts_accuracy(num_samples=100, mcts_iterations=800)

# TODO: evaluate using "search", rather than "predict" to get the best move from MCTS.
# evaluate_model_soft_accuracy(num_samples=100, threshold=0.01, mcts_iterations=500)


# evaluate_supervised_mcts_on_test_data(
#     num_samples=300, mcts_iterations=800
# )