from util.supervised_mcts import SupervisedMCTS, evaluate_supervised_mcts_on_test_data
from games.connect4 import Connect4
import torch
import numpy as np
from util.data_transformer import (
    DataTransformer,
    convert_board_to_tensor,
    convert_tensor_to_board,
)
import os
from networks.Connect4Net import Connect4Net


# Load the model / initialize the model
model_path = "models/connect4_4x4_alpha_zero_50k.pt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Connect4Net().to(device)
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))

# Load the full dataset that was generated earlier.
data_path = "data/connect4_4x4_training_data_50k.npy"
transformer = DataTransformer(data_path, batch_size=1)

# 40,000 train data samples. Only using boards, not labels.
boards, _, _ = transformer.get_evaluation_data(type="train")


# mcts parameters
iterations = 800
exploration_constant = 1.5

# alpha zero parameters
batch_size = 100
num_epochs = 1

training_data = []

# Complete Games:  AlphaZero plays each training game to completion (win, loss, or draw) during the self-play data generation phase.

# Ground Truth Values: For each position encountered during these games, the actual outcome is recorded (+1 for win, -1 for loss, 0 for draw from the perspective of the player to move).

supervised_mcts = SupervisedMCTS(
    model=model,
    iterations=iterations,
    exploration_constant=exploration_constant,
    selection_method="PUCT",
    method="AlphaZero",
)

for i, episode in enumerate(boards):

    game = Connect4(num_of_rows=4, num_of_cols=4, board=episode)

    episode_states = []
    batch = []
    value = None
    while value is None:
        policy = supervised_mcts.search(game)

        # add policy and state to episode states
        episode_states.append((game.board.copy(), policy.copy()))

        best_move = np.argmax(policy)

        game.make_move(best_move)

        value = game.evaluate_board()

    winner = 1
    for state in reversed(episode_states):
        board, policy = state
        # add to training data
        training_data.append((board, policy, value * winner))

        batch.append((board, policy, value * winner))
        winner *= -1
    break

    if len(batch) >= batch_size:
        # train the model
        model.train_on_batch(batch, num_epochs)
        batch = []
        print(f"Trained on {len(training_data)} samples.")

    if i % 100 == 0:
        print(
            f"Episode {i+1}/{len(boards)} completed. Total training data: {len(training_data)}"
        )

    # if len(training_data) >= 2000:
    #     break

    # if i > 1_000:
    #     break
    if len(training_data) >= 400:
        break

print(f"Total training data: {len(training_data)}")

# save the training data
training_data_array = np.array(training_data, dtype=object)
np.save("data/connect4_4x4_self_play_data_50k.npy", training_data_array)
print("Training data saved to data/connect4_4x4_training_data_50k.npy")

# save the model using model_path
save_path = "models/connect4_4x4_alpha_zero_50k.pt"
os.makedirs(os.path.dirname(save_path), exist_ok=True)
torch.save(model.state_dict(), save_path)
print(f"Model saved to {save_path}")


# for i, case in enumerate(cases):
#     board = case

#     game = Connect4(num_of_rows=4, num_of_cols=4, board=board)

#     supervised_mcts = SupervisedMCTS(
#         model=model,
#         iterations=iterations,
#         exploration_constant=exploration_constant,
#         selection_method="PUCT",
#         method="AlphaZero",
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


# # evaluate on test data
# accuracy = evaluate_supervised_mcts_on_test_data(
#     num_samples=1_000,
#     mcts_iterations=iterations,
#     exploration_constant=exploration_constant,
#     selection_method="PUCT",
# )
