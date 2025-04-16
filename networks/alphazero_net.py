import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """
    A simple Residual Block with two convolution layers.
    """

    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)


class AlphaZeroNet(nn.Module):
    def __init__(
        self, board_size=(4, 4), num_channels=128, num_res_blocks=3, num_moves=4
    ):
        """
        board_size: tuple with (rows, cols) (for our case 4x4)
        num_channels: number of filters in the convolution layers
        num_res_blocks: how many residual blocks to use
        num_moves: number of legal moves, here 4 for 4x4 Connect4.
        """
        super().__init__()
        self.board_rows, self.board_cols = board_size

        # Initial convolution layer
        self.initial_conv = nn.Sequential(
            nn.Conv2d(2, num_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_channels),
            nn.ReLU(),
        )

        # Residual blocks
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(num_channels) for _ in range(num_res_blocks)]
        )

        # Policy head: one convolution, flatten, and FC layer to output logits.
        self.policy_conv = nn.Sequential(
            nn.Conv2d(num_channels, 2, kernel_size=1),
            nn.BatchNorm2d(2),
            nn.ReLU(),
        )
        self.policy_fc = nn.Linear(2 * self.board_rows * self.board_cols, num_moves)

        # Value head: one convolution, FC layers, ending with tanh.
        self.value_conv = nn.Sequential(
            nn.Conv2d(num_channels, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
        )
        self.value_fc1 = nn.Linear(self.board_rows * self.board_cols, 256)
        self.value_fc2 = nn.Linear(256, 1)

    def forward(self, x):
        """
        x: input tensor of shape [B, 2, rows, cols]
        """
        out = self.initial_conv(x)
        out = self.res_blocks(out)

        # Policy head forward
        policy = self.policy_conv(out)
        policy = policy.view(policy.size(0), -1)  # Flatten
        policy_logits = self.policy_fc(policy)

        # Value head forward
        value = self.value_conv(out)
        value = value.view(value.size(0), -1)  # Flatten
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))

        return policy_logits, value
