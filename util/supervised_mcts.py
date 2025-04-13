import random
import math
import numpy as np
import copy
import torch
import torch.nn as nn
from games.connect4 import Connect4
from util.solver import Solver
from util.data_transformer import DataTransformer
from networks.Connect4Net import Connect4Net
from tqdm import tqdm

# Set seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)


def clone_game(game: Connect4) -> Connect4:
    """
    self explanatory, just making a copy of the game.
    """
    new_board = np.copy(game.board)
    new_game = Connect4(
        board=new_board,
        num_of_rows=game.num_of_rows,
        num_of_cols=game.num_of_cols,
    )
    new_game.move_count = game.move_count
    new_game.move_history = game.move_history.copy()
    new_game.result = game.result
    return new_game


class MCTSNode:
    def __init__(self, game: Connect4, parent=None, move=None):
        self.game = game  # a clone of the Connect4 game state
        self.parent = parent
        self.move = move  # the move that led to this node (None for the root)
        self.children = {}  # dictionary: move (int) -> child node
        self.visits = 0
        self.wins = 0.0  # cumulative reward (from the root's perspective)
        self.untried_moves = game.get_legal_moves()  # moves that have not yet been expanded

    def is_fully_expanded(self):
        return len(self.untried_moves) == 0

    def best_child(self, exploration_constant=1.41):
        """
        self explanatory, select the child with the highest UCT value.
        """
        best_score = -float("inf")
        best_child = None
        for move, child in self.children.items():
            score = (child.wins / child.visits) + exploration_constant * math.sqrt(math.log(self.visits) / child.visits)
            if score > best_score:
                best_score = score
                best_child = child
        return best_child


class SupervisedMCTS:
    def __init__(self, model_path, iterations=1000, exploration_constant=1.41, device=None):
        self.iterations = iterations
        self.exploration_constant = exploration_constant
        self.samples = []  # will store tuples of (board state, outcome)
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = Connect4Net().to(self.device) #loading
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

    def convert_board_to_tensor(self, board):
        """
        convert the board (numpy array with values -1, 0, 1) to a tensor of shape [2, rows, cols]:
        channel 0 for player 1's pieces, channel 1 for player -1's pieces.
        """
        board_tensor = np.zeros((2, board.shape[0], board.shape[1]), dtype=np.float32)
        board_tensor[0] = (board == 1).astype(np.float32)
        board_tensor[1] = (board == -1).astype(np.float32)
        return torch.tensor(board_tensor, device=self.device)

    def evaluate_with_model(self, game: Connect4) -> float:
        """
        evaluate the current game state using the trained model.
        returns the value prediction (expected outcome) as a float in [-1, 1].
        """
        board = game.board
        board_tensor = self.convert_board_to_tensor(board).unsqueeze(0)  # add batch dimension
        with torch.no_grad():
            _, value = self.model(board_tensor)
        return value.item()

    def rollout(self, game: Connect4, root_player: int) -> float:
        """
        Evaluate the leaf node.
        If the game is terminal, return a fixed reward:
            +1.0 for a win, -1.0 for a loss, 0.0 for a draw.
        Otherwise, return the network's value prediction.
        """
        result = game.evaluate_board()
        if result is not None:
            if result > 0:
                winner = 1
            elif result < 0:
                winner = -1
            else:
                winner = 0

            if winner == root_player:
                return 1.0
            elif winner == 0:
                return 0.0
            else:
                return -1.0

        # For a non-terminal state, use the model's prediction.
        value = self.evaluate_with_model(game)
        return value

    def search(self, game: Connect4) -> np.ndarray:
        """
        Run MCTS simulations from the given game state using the trained model at the leaf nodes.
        Returns a probability vector over moves (for every column) derived from the visit counts at the root.
        """
        root_player = 1
        root_node = MCTSNode(clone_game(game))

        for _ in range(self.iterations):
            node = root_node
            simulation_game = clone_game(game)

            #selection process
            while node.is_fully_expanded() and node.children:
                node = node.best_child(self.exploration_constant)
                simulation_game.make_move(node.move)

            #expansion process
            if node.untried_moves:
                move = random.choice(node.untried_moves)
                simulation_game.make_move(move)
                node.untried_moves.remove(move)
                child_node = MCTSNode(clone_game(simulation_game), parent=node, move=move)
                node.children[move] = child_node
                node = child_node

            #save the board state from where the evaluation will start.
            sample_state = np.copy(simulation_game.board)

            #evaluation using the Model instead of rollout as we discussed.
            reward = self.rollout(clone_game(simulation_game), root_player)

            #save the sample
            self.samples.append((sample_state, reward))

            #backpropogation
            current_reward = reward
            while node is not None:
                node.visits += 1
                node.wins += current_reward
                current_reward = -current_reward  # flip reward for the opponent
                node = node.parent

        #compute move probabilities for the root node using normalized visit counts.
        move_visits = {move: child.visits for move, child in root_node.children.items()}
        total_visits = sum(move_visits.values())
        move_avg_scores = {move: (child.wins / child.visits) for move, child in root_node.children.items() if child.visits > 0}

        # Normalize average scores to a common scale (e.g., rescale to [0,1] if possible).
        # For example, if values are in [-1, 1]:
        normalized_avg = {move: (score + 1) / 2 for move, score in move_avg_scores.items()}

        # Compute a combined score. Let alpha weight the average value.
        alpha = 0.5
        combined_score = {}
        for move in move_visits:
            visits_score = move_visits[move] / total_visits
            value_score = normalized_avg.get(move, 0.0)
            combined_score[move] = alpha * value_score + (1 - alpha) * visits_score

        # Create probability distribution via softmax (or simply pick the max).
        scores = np.array([combined_score.get(move, 0) for move in range(game.num_of_cols)])
        temperature = 1.0
        exp_scores = np.exp(scores / temperature)  # temperature can be tuned
        if exp_scores.sum() > 0:
            move_probs = exp_scores / exp_scores.sum()
        else:
            move_probs = np.zeros_like(scores)
        return move_probs



        move_visits = {move: child.visits for move, child in root_node.children.items()}
        total_visits = sum(move_visits.values())
        move_probs = np.zeros(game.num_of_cols)
        for move, visits in move_visits.items():
            move_probs[move] = visits / total_visits
        return move_probs

    def predict(self, game: Connect4):
        """
        Evaluate the current game state using the trained model.
        Returns:
            - policy: A probability distribution over moves (numpy array).
            - value: A scalar value prediction in [-1, 1].
        """
        board_tensor = self.convert_board_to_tensor(game.board).unsqueeze(0)  # add batch dimension
        with torch.no_grad():
            policy_logits, value = self.model(board_tensor)
        # Convert logits to probabilities
        policy = torch.softmax(policy_logits, dim=1).cpu().numpy().flatten()
        return policy, value.item()

    def get_samples(self):
        """
        Retrieve the board samples and outcomes collected during MCTS search.
        """
        return self.samples


############################### Evaluation Routine ################################


def evaluate_supervised_mcts_accuracy(num_samples=100, mcts_iterations=500):
    """
    Evaluate SupervisedMCTS accuracy on random nonterminal board positions.
    For each sampled board position, we compare the best move from the Solver (ground truth)
    to the best move from SupervisedMCTS.
    """
    correct = 0
    total = 0

    model_path = "models/connect4_4x4_supervised.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mcts = SupervisedMCTS(model_path=model_path, iterations=mcts_iterations, device=device)
    solver_accuracy = Solver(Connect4(num_of_rows=4, num_of_cols=4))

    for _ in tqdm(range(num_samples), desc="Evaluating random positions", unit="game"):
        game = Connect4(num_of_rows=4, num_of_cols=4)
        # this a random number of moves (between 0 and 5) ensuring game is not terminal
        num_moves = random.randint(0, 5)
        for _ in range(num_moves):
            legal = game.get_legal_moves()
            if not legal:
                break
            move = random.choice(legal)
            game.make_move(move)
            if game.evaluate_board() is not None:
                break

        if game.evaluate_board() is not None:
            continue

        # use the Solver to get the ground truth best move.
        policy_solver, _ = solver_accuracy.evaluate_state()
        # best_move_solver = int(np.argmax(policy_solver))

        # there may be multiple best moves, we find all best moves
        best_moves_solver = []
        best_value = np.max(policy_solver)
        for move in range(len(policy_solver)):
            # check if move is a best move
            if best_value == policy_solver[move]:
                best_moves_solver.append(move)

        # use SupervisedMCTS to get the best move.
        move_probs = mcts.search(clone_game(game))
        best_move_mcts = int(np.argmax(move_probs))

        # compare moves.
        if best_move_mcts in best_moves_solver:
            correct += 1
        total += 1

    accuracy = (correct / total) * 100 if total > 0 else 0.0
    print(f"SupervisedMCTS Accuracy: {accuracy:.2f}% over {total} evaluated positions.")


def evaluate_model_soft_accuracy(num_samples=100, threshold=0.01, mcts_iterations=500):
    """
    Evaluate the model's soft policy accuracy over a set of random nonterminal board positions.
    For each board:
      - Use the trained model (via SupervisedMCTS.predict) to get a predicted policy.
      - Use the Solver (minimax) to get the target policy.
    A prediction is considered "good" if the predicted move's probability is within
    `threshold` of the maximum target probability.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "models/connect4_4x4_supervised.pt"
    mcts = SupervisedMCTS(model_path=model_path, iterations=mcts_iterations, device=device)
    
    good_enough = 0
    total = 0

    for _ in range(num_samples):
        game = Connect4(num_of_rows=4, num_of_cols=4)
        num_moves = random.randint(0, 5)
        for _ in range(num_moves):
            legal_moves = game.get_legal_moves()
            if not legal_moves:
                break
            move = random.choice(legal_moves)
            game.make_move(move)
            # stop if the game becomes terminal.
            if game.evaluate_board() is not None:
                break

        # skip terminal positions.
        if game.evaluate_board() is not None:
            continue

        # use the model to predict the policy.
        pred_policy, _ = mcts.predict(game)
        solver = Solver(game)
        target_policy, _ = solver.evaluate_state()

        # consider the prediction "good" if the predicted move (argmax of pred_policy)
        # has a target probability within threshold of the maximum target probability.
        pred_move = int(np.argmax(pred_policy))
        max_target = np.max(target_policy)
        if target_policy[pred_move] >= (max_target - threshold):
            good_enough += 1
        total += 1

    if total > 0:
        accuracy = (good_enough / total) * 100
        print(f"Soft Policy Accuracy: {accuracy:.2f}% over {total} samples.")
    else:
        print("No nonterminal samples available for evaluation.")


def evaluate_supervised_mcts_on_test_data(
    num_samples=None, mcts_iterations=800
):
    """
    Evaluate SupervisedMCTS on the held-out 20% test data from the generated training set.
    For each board in the test set:
      - Use SupervisedMCTS.predict to get the predicted policy.
      - Compare the predicted best move to the target best move from the training data.
    A prediction is considered correct if the predicted move's target probability is within
    `threshold` of the maximum target probability.
    """
    # Load the full dataset that was generated earlier.
    data_path = "data/connect4_4x4_training_data.npy"
    transformer = DataTransformer(data_path, batch_size=1)

    # 2,000 test data samples (20% test from original 10,000 samples)
    # to guarantee we are evaluating on data we haven't trained on.
    eval_boards, eval_target_policy, eval_target_value = (
        transformer.get_evaluation_data(type="test")
    )

    # if num_samples is provided, sample that many test examples.
    if num_samples is None:
        num_samples = len(eval_boards)
    else:
        num_samples = min(num_samples, len(eval_boards))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "models/connect4_4x4_supervised.pt"
    mcts = SupervisedMCTS(
        model_path=model_path, iterations=mcts_iterations, device=device
    )

    correct = 0
    total = 0

    for i in tqdm(range(num_samples), desc="Evaluating MCTS", unit="board"):
        board = eval_boards[i]
        target_policy = eval_target_policy[i]
        target_value = eval_target_value[i]

        game = Connect4(num_of_rows=4, num_of_cols=4, board=board)
        # pred_policy, _ = mcts.predict(game)
        pred_policy = mcts.search(clone_game(game))
        max_pred = np.max(pred_policy)
        pred_move = np.argmax(pred_policy)
        # print("predicted move:", pred_move, "with value:", max_pred)

        policy_moves = []
        best_policy_value = np.max(target_policy)
        for j in range(len(target_policy)):
            if best_policy_value == target_policy[j]:
                policy_moves.append(j)

        # Consider the prediction correct if the predicted move is any of the best moves
        if pred_move in policy_moves:
            correct += 1
        total += 1

    accuracy = (correct / total) * 100 if total > 0 else 0.0
    print(f"SupervisedMCTS Test Data Accuracy: {accuracy:.2f}% over {total} samples.")


# evaluate_supervised_mcts_accuracy(num_samples=100, mcts_iterations=800)

# TODO: evaluate using "search", rather than "predict" to get the best move from MCTS.
# evaluate_model_soft_accuracy(num_samples=100, threshold=0.01, mcts_iterations=500)


evaluate_supervised_mcts_on_test_data(
    num_samples=300, mcts_iterations=2000
)
