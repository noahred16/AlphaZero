import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import os
from util.data_transformer import DataTransformer
from networks.Connect4Net import Connect4Net


# Set seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the training data (10,000 randomly generated samples with labeled policy and value)
data_path = "data/connect4_4x4_training_data_100k.npy"

############################### Prepare Your Dataset ###############################
data_transformer = DataTransformer(data_path)

# Dataloaders
train_loader = data_transformer.get_training_data()
test_loader = data_transformer.get_testing_data()

print(f"Training dataset size: {len(train_loader.dataset)}")
print(f"Testing dataset size: {len(test_loader.dataset)}")

############################### Training Loop ###############################
model = Connect4Net().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
policy_loss_fn = nn.CrossEntropyLoss()
value_loss_fn = nn.MSELoss()

EPOCHS = 10

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for boards, target_policy, target_value in train_loader:
        optimizer.zero_grad()

        pred_policy_logits, pred_value = model(boards)

        # Policy: target is a probability distribution (soft labels), so use log_softmax + KLDivLoss or soft cross-entropy
        policy_log_probs = torch.log_softmax(pred_policy_logits, dim=1)
        policy_loss = torch.sum(-target_policy * policy_log_probs) / target_policy.size(
            0
        )

        # Value: simple MSE
        value_loss = value_loss_fn(pred_value, target_value)

        loss = policy_loss + value_loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1} | Loss: {total_loss:.4f}")


############################### Save the Trained Model ###############################
# Save the model
save_path = "models/connect4_4x4_supervised_100k.pt"
os.makedirs(os.path.dirname(save_path), exist_ok=True)
torch.save(model.state_dict(), save_path)
print(f"Model saved to {save_path}")


############################### Evaluate ###############################
def evaluate_soft(model, loader, threshold=0.01):
    model.eval()
    with torch.no_grad():
        total = 0
        good_enough = 0
        for boards, target_policy, _ in loader:
            logits, _ = model(boards)
            probs = torch.softmax(logits, dim=1)  # [B, 4]

            # Check if predicted column is within threshold of max target value
            for i in range(probs.size(0)):
                pred_col = torch.argmax(probs[i])
                target_probs = target_policy[i]
                max_val = torch.max(target_probs)
                if target_probs[pred_col] >= (max_val - threshold):
                    good_enough += 1
                total += 1
        print(
            f"Soft Policy Accuracy (within {threshold:.2f} of max): {good_enough}/{total} = {good_enough/total:.2%}"
        )


evaluate_soft(model, test_loader, threshold=0.01)


############################### Demo Policy ###############################
# from games.connect4 import Connect4
# from util.solver import Solver

# game = Connect4(num_of_rows=4, num_of_cols=4)
# model.eval()  # Set the model to evaluation mode

# while True:
#     game.print_pretty()
#     print("Current board state:")
#     # print(game.board)

#     # Convert the current board to tensor
#     board_tensor = convert_board_to_tensor(game.board).unsqueeze(
#         0
#     )  # Add batch dimension

#     with torch.no_grad():
#         pred_policy_logits, pred_value = model(board_tensor.to(device))
#         pred_policy = torch.softmax(pred_policy_logits, dim=1).cpu().numpy().flatten()
#         pred_value = pred_value.cpu().item()

#     # Find the best move based on the policy
#     best_move = np.argmax(pred_policy)
#     legal_moves = game.get_legal_moves()

#     if best_move not in legal_moves:
#         print("Best move is not legal, choosing random legal move.")
#         best_move = np.random.choice(legal_moves)

#     # compare with solver
#     solver = Solver(game)
#     policy, value = solver.evaluate_state()

#     print(
#         f"AI chooses column {best_move} "
#         f"with policy: {np.round(pred_policy, 3)}, "
#         f"value: {pred_value:.2f}"
#     )
#     print(f"Solver policy: {policy}, value: {value:.2f}")
#     print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

#     # Make the move
#     game.make_move(best_move)

#     game.evaluate_board()
#     if game.result is not None:
#         print(f"Game Over! Result: {game.result}")
#         break
