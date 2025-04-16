import torch.nn as nn
import torch
import torch.optim as optim


class Connect4Net(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(2, 128, kernel_size=3, padding=1),  # [B, 128, 4, 4]
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.flatten = nn.Flatten()

        self.fc = nn.Sequential(
            nn.Linear(128 * 4 * 4, 256),  # was 128 before
            nn.ReLU(),
            nn.Dropout(0.1),  # low dropout just in case
            nn.Linear(256, 128),  # new additional FC layer
            nn.ReLU(),
        )

        self.policy_head = nn.Linear(128, 4)
        self.value_head = nn.Sequential(nn.Linear(128, 1), nn.Tanh())

    def forward(self, x):
        x = self.conv(x)
        x = self.flatten(x)
        x = self.fc(x)
        policy = self.policy_head(x)
        value = self.value_head(x)
        return policy, value

    def train_on_batch(self, batch, epochs=1):
        self.train()
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        policy_loss_fn = nn.CrossEntropyLoss()
        value_loss_fn = nn.MSELoss()

        for epoch in range(epochs):
            total_loss = 0
            for boards, target_policy, target_value in batch:
                optimizer.zero_grad()

                pred_policy_logits, pred_value = self(boards)

                # Policy: target is a probability distribution (soft labels), so use log_softmax + KLDivLoss or soft cross-entropy
                policy_log_probs = torch.log_softmax(pred_policy_logits, dim=1)
                policy_loss = torch.sum(
                    -target_policy * policy_log_probs
                ) / target_policy.size(0)

                # Value: simple MSE
                value_loss = value_loss_fn(pred_value, target_value)

                loss = policy_loss + value_loss
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            print(f"Epoch {epoch+1} | Loss: {total_loss:.4f}")
