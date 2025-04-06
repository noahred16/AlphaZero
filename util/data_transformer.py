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


def convert_tensor_to_board(tensor):
    # tensor.shape: torch.Size([1, 2, 4, 4])
    tensor = tensor.cpu().numpy()

    board = np.zeros((tensor.shape[2], tensor.shape[3]), dtype=np.int8)

    board[tensor[0, 0] > 0] = 1  # Player 1 - first channel
    board[tensor[0, 1] > 0] = -1  # Player -1 - second channel

    return board


class DataTransformer:
    def __init__(self, data_path, batch_size=64):
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
        self.train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size)

    def get_training_data(self):
        return self.train_loader

    def get_testing_data(self):
        return self.test_loader

    def get_evaluation_data(self, type="test"):
        """
        This class by default converts the data into tensors for training.
        For mcts evaluation we need it formated as numpy arrays to pass to the connect4 game.
        """
        loader = self.test_loader if type == "test" else self.train_loader

        eval_boards = []
        eval_target_policy = []
        eval_target_value = []
        for board, target_policy, target_value in loader:
            eval_boards.append(convert_tensor_to_board(board))
            eval_target_policy.append(target_policy.cpu().numpy()[0])
            eval_target_value.append(target_value.cpu().numpy()[0])
        return eval_boards, eval_target_policy, eval_target_value
