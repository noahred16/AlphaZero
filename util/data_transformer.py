import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split

np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def convert_board_to_tensor(board):
    board_tensor = np.zeros((2, board.shape[0], board.shape[1]), dtype=np.float32)
    board_tensor[0] = (board == 1).astype(np.float32)  # Player 1
    board_tensor[1] = (board == -1).astype(np.float32)  # Player -1
    return torch.tensor(board_tensor, device=device)


class DataTransformer:
    def __init__(self, data_path):
        training_data = np.load(data_path, allow_pickle=True)

        boards = []
        policies = []
        values = []

        for board, policy, value in training_data:
            boards.append(convert_board_to_tensor(board))
            policies.append(
                torch.tensor(policy, dtype=torch.float32, device=device)
            )  # [4] for 4 columns
            values.append(
                torch.tensor([value], dtype=torch.float32, device=device)
            )  # [1] scalar

        # Stack tensors
        boards = torch.stack(boards)  # Shape: [N, 2, 4, 4]
        policies = torch.stack(policies)  # Shape: [N, 4]
        values = torch.stack(values)  # Shape: [N, 1]

        # Create a dataset
        dataset = TensorDataset(boards, policies, values)

        # Split into train/test (80/20)
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

        # Dataloaders
        self.train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=64)

    def get_training_data(self):
        return self.train_loader

    def get_testing_data(self):
        return self.test_loader
