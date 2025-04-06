import torch.nn as nn


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
