import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

from util.data_transformer import DataTransformer
from networks.alphazero_net import AlphaZeroNet

# Set seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the training data using the DataTransformer (the same data as before)
data_path = "data/connect4_4x4_training_data.npy"
data_transformer = DataTransformer(data_path)
train_loader = data_transformer.get_training_data()
test_loader = data_transformer.get_testing_data()

print(f"Training dataset size: {len(train_loader.dataset)}")
print(f"Testing dataset size: {len(test_loader.dataset)}")

# Instantiate the AlphaZero network
model = AlphaZeroNet(board_size=(4, 4), num_channels=128, num_res_blocks=3, num_moves=4).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Loss functions:
# For policy: using soft labels and log softmax (this is analogous to soft cross entropy / KL divergence)
# For value: MSE loss
def train_epoch(model, optimizer, loader):
    model.train()
    total_loss = 0
    for boards, target_policy, target_value in loader:
        optimizer.zero_grad()
        #forward pass
        policy_logits, pred_value = model(boards)
        #policy loss: using negative log likelihood on soft targets
        policy_log_probs = torch.log_softmax(policy_logits, dim=1)
        policy_loss = -torch.sum(target_policy * policy_log_probs) / target_policy.size(0)
        #value loss using MSE
        value_loss = nn.MSELoss()(pred_value, target_value)
        loss = policy_loss + value_loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss

def evaluate_soft(model, loader, threshold=0.01):
    model.eval()
    with torch.no_grad():
        total = 0
        correct = 0
        for boards, target_policy, _ in loader:
            policy_logits, _ = model(boards)
            probs = torch.softmax(policy_logits, dim=1)
            for i in range(probs.size(0)):
                pred_move = torch.argmax(probs[i]).item()
                max_target = torch.max(target_policy[i]).item()
                #prediction is "good" if the probability for the chosen move is within the threshold of the maximum target probability.
                if target_policy[i][pred_move] >= (max_target - threshold):
                    correct += 1
                total += 1
    acc = correct / total if total > 0 else 0
    print(f"Soft Policy Accuracy: {correct}/{total} = {acc*100:.2f}%")
    return acc

#training loop
EPOCHS = 10
for epoch in range(EPOCHS):
    loss = train_epoch(model, optimizer, train_loader)
    print(f"Epoch {epoch+1} | Loss: {loss:.4f}")
    evaluate_soft(model, test_loader, threshold=0.01)

#save the trained model
save_path = "models/alphazero_4x4.pt"
os.makedirs(os.path.dirname(save_path), exist_ok=True)
torch.save(model.state_dict(), save_path)
print(f"Model saved to {save_path}")
