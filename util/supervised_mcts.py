import random
import math
import numpy as np
import copy
import torch
import torch.nn as nn
from games.connect4 import Connect4
from util.solver import Solver
from util.data_transformer import DataTransformer, convert_board_to_tensor
from networks.Connect4Net import Connect4Net
from tqdm import tqdm
import graphviz
import io
from contextlib import redirect_stdout

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
        self.num_visits = 0
        self.total_score = 0  # cumulative reward (from the root's perspective)
        self.untried_moves = (
            game.get_legal_moves()
        )  # moves that have not yet been expanded
        self.ucb_score = None
        if self.parent is not None:
            self.turn = -self.parent.turn
        else:
            self.turn = 1
        self.policy_priors = None  # for PUCT selection method

    def is_fully_expanded(self):
        return len(self.untried_moves) == 0

    def is_terminal(self):
        return self.game.evaluate_board() is not None

    def get_ucb(self, exploration_constant):
        if self.num_visits == 0:
            score = float("inf")
            self.ucb_score = score
            return score

        exploitation = (self.total_score / self.num_visits) * -1 * self.turn

        exploration = exploration_constant * math.sqrt(
            math.log(self.parent.num_visits) / self.num_visits
        )

        score = exploitation + exploration
        self.ucb_score = score
        return score

    def get_puct(self, exploration_constant, policy_priors):
        """
        PUCT formula:
            Q(s,a) + c_puct * P(s,a) * sqrt(sum_b N(s,b)) / (1 + N(s,a))
        """
        if self.num_visits == 0:
            score = float("inf")
            self.ucb_score = score
            return score

        exploitation = (self.total_score / self.num_visits) * -1 * self.turn

        c_puct = exploration_constant

        exploration = (
            c_puct
            * policy_priors[self.move]
            * math.sqrt(self.parent.num_visits)
            / (1 + self.num_visits)
        )

        score = exploitation + exploration
        return score

    def best_child(self, exploration_param, selection_method, policy_priors=None):
        best_node = None
        best_value = float("-inf")

        vals = {}
        num_visits = {}
        for move, child in self.children.items():

            if selection_method == "UCB-1":
                value = child.get_ucb(exploration_param)
            elif selection_method == "PUCT":
                value = child.get_puct(exploration_param, policy_priors)

            vals[move] = value
            num_visits[move] = child.num_visits

            if value > best_value:
                best_value = value
                best_node = child

        return best_node


class SupervisedMCTS:
    def __init__(
        self,
        model,
        iterations=1000,
        exploration_constant=1,
        selection_method="UCB-1",  # "PUCT" or "UCB-1"
        method="Standard",  # "Standard" or "AlphaZero"
    ):
        self.iterations = iterations
        self.exploration_constant = exploration_constant
        self.selection_method = selection_method
        self.method = method

        self.samples = []  # will store tuples of (board state, outcome)

        self.model = model
        self.model.eval()

        self.root = None

    def tree_visualization(self):
        def add_node(node: MCTSNode, graph: graphviz.Digraph):
            node_id = str(id(node))

            # Capture the output of print_pretty()
            output = io.StringIO()
            with redirect_stdout(output):
                # deep copy the board
                copy = node.game.board.copy()
                # flip the board for display
                copy = copy * node.turn
                game = Connect4(
                    board=copy,
                    num_of_rows=node.game.num_of_rows,
                    num_of_cols=node.game.num_of_cols,
                )
                game.print_pretty()

            game_display = output.getvalue()

            ucb = round(node.ucb_score, 2) if node.ucb_score is not None else "N/A"

            exploration = (
                round(
                    self.exploration_constant
                    * math.sqrt(math.log(node.parent.num_visits) / node.num_visits),
                    3,
                )
                if node.parent is not None and node.num_visits > 0
                else "N/A"
            )

            # Format the label with statistics and game state
            label = f"Move: {node.move if node.move is not None else 'Root'}\nTotal Score: {node.total_score:.2f}\nVisits: {node.num_visits}\nAvg Score: {node.total_score / node.num_visits:.2f}\nExploration: {exploration}\nUCB: {ucb}\n Player: {'X' if node.turn == 1 else 'O'}\n"
            label += game_display

            # Replace newlines with \n for proper display in graphviz
            label = label.replace("\n", "\\n")

            graph.node(node_id, label=label)

            if node.parent is not None:
                graph.edge(str(id(node.parent)), node_id, label=str(node.move))

            for move in sorted(node.children.keys()):
                child = node.children[move]
                add_node(child, graph)

        # Create a graph with a larger node size to accommodate the game display
        graph = graphviz.Digraph(format="png")
        graph.attr(
            "node", shape="box", fontname="Courier New"
        )  # Monospace font for game display

        add_node(self.root, graph)

        file_name = "mcts_tree"
        graph.render(file_name, cleanup=True)
        print(f"Tree visualization saved as {file_name}.png")

    def evaluate_with_model(self, game: Connect4) -> float:
        """
        evaluate the current game state using the trained model.
        returns the value prediction (expected outcome) as a float in [-1, 1].
        """
        board = game.board
        board_tensor = convert_board_to_tensor(board).unsqueeze(0)
        with torch.no_grad():
            _, value = self.model(board_tensor)
        return value.item()

    def evaluate_policy_with_model(self, game: Connect4) -> np.ndarray:
        """
        evaluate the current game state using the trained model.
        returns the policy prediction as a probability distribution over moves.
        """
        board = game.board
        board_tensor = convert_board_to_tensor(board).unsqueeze(0)
        with torch.no_grad():
            policy_logits, _ = self.model(board_tensor)
        policy = torch.softmax(policy_logits, dim=1).cpu().numpy().flatten()
        return policy

    def selection(self, node: MCTSNode):
        while node.is_fully_expanded() and not node.is_terminal():
            if node.policy_priors is None and self.selection_method == "PUCT":
                node.policy_priors = self.evaluate_policy_with_model(node.game)
            node = node.best_child(
                exploration_param=self.exploration_constant,
                selection_method=self.selection_method,
                policy_priors=node.policy_priors,
            )
        return node

    def expansion(self, node: MCTSNode):
        # no need to expand if the leaf is terminal
        # no expansion if
        if node.is_terminal():
            return node

        # for standard MCTS, we don't expand on first visit but we do for AlphaZero.
        if node.num_visits == 0 and self.method != "AlphaZero":
            return node

        action = random.choice(node.untried_moves)
        node.untried_moves.remove(action)

        new_game = clone_game(node.game)
        new_game.make_move(action)

        child_node = MCTSNode(new_game, parent=node, move=action)
        node.children[action] = child_node
        return child_node

    def rollout(self, node: MCTSNode):
        # no need to "rollout" if the node is terminal
        if node.is_terminal():
            result = node.game.evaluate_board() * node.turn * -1
            return result
            # no scaling, but performs worse
            # return 1.0 if result > 0 else -1.0 if result < 0 else 0.0
        prediction = self.evaluate_with_model(node.game) * node.turn
        return prediction
        # no scaling, but performs worse
        # return 1.0 if prediction > 0 else -1.0 if prediction < 0 else 0.0

    def backpropagation(self, node: MCTSNode, result: float):
        while node is not None:
            node.num_visits += 1
            node.total_score += result
            node = node.parent

    # supervised mcts
    def search(self, game: Connect4) -> np.ndarray:
        """
        Run MCTS simulations from the given game state using the trained model at the leaf nodes.
        Returns a probability vector over moves (for every column) derived from the visit counts at the root.
        """
        root_node = MCTSNode(clone_game(game))
        self.root = root_node

        for _ in range(self.iterations):
            # 1.) SELECTION
            leaf = self.selection(root_node)

            # 2.) EXPANSION
            child = self.expansion(leaf)

            # 3.) SIMULATION
            result = self.rollout(child)

            # 4.) BACKPROPAGATION
            self.backpropagation(child, result)

        # Compute move probabilities based on visit counts
        move_visits = {
            move: child.num_visits for move, child in root_node.children.items()
        }
        total_visits = sum(move_visits.values())
        move_probs = np.zeros(game.num_of_cols)
        for move, visits in move_visits.items():
            move_probs[move] = visits / total_visits if total_visits > 0 else 0.0
        return move_probs

    def predict(self, game: Connect4):
        """
        Evaluate the current game state using the trained model.
        Returns:
            - policy: A probability distribution over moves (numpy array).
            - value: A scalar value prediction in [-1, 1].
        """
        board_tensor = convert_board_to_tensor(game.board).unsqueeze(
            0
        )  # add batch dimension
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

    model_path = "models/connect4_4x4_supervised_50k.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Connect4Net().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    mcts = SupervisedMCTS(model=model, iterations=mcts_iterations)
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


def evaluate_supervised_mcts_on_test_data(
    exploration_constant,
    model,
    num_samples=None,
    mcts_iterations=800,
    selection_method="UCB-1",
    method="Standard",
):
    """
    Evaluate SupervisedMCTS on the held-out 20% test data from the generated training set.
    For each board in the test set:
      - Use SupervisedMCTS.predict to get the predicted policy.
      - Compare the predicted best move to the target best move from the training data.
    A prediction is considered correct if the predicted move's target probability is within
    `threshold` of the maximum target probability.
    """

    print(
        f"Evaluating SupervisedMCTS on test data with exploration constant: {exploration_constant}, and iterations: {mcts_iterations}"
    )
    # Load the full dataset that was generated earlier.
    data_path = "data/connect4_4x4_training_data_50k.npy"
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

    # model_path = "models/connect4_4x4_supervised_50k.pt"
    # if method == "AlphaZero":
    #     model_path = "models/connect4_4x4_alpha_zero_50k.pt"
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = Connect4Net().to(device)
    # model.load_state_dict(torch.load(model_path, map_location=device))

    correct = 0
    incorrect = 0
    total = 0

    for i in tqdm(range(num_samples), desc="Evaluating MCTS", unit="board"):

        mcts = SupervisedMCTS(
            model=model,
            iterations=mcts_iterations,
            exploration_constant=exploration_constant,
            selection_method=selection_method,
            method=method,
        )

        board = eval_boards[i]
        target_policy = eval_target_policy[i]
        target_value = eval_target_value[i]

        # Skip over test cases where all actions are eq
        # if len(target_policy) == 4 

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

        # Skip over test cases where all actions are equal
        if len(policy_moves) == len(target_policy):
            continue

        # Consider the prediction correct if the predicted move is any of the best moves
        if pred_move in policy_moves:
            correct += 1
        else:
            game.print_pretty()
            print(
                f"Predicted move: {pred_move} with value: {max_pred}, but best moves were: {policy_moves}"
            )
            print(f"Policy: {pred_policy}")
            incorrect += 1
        total += 1

    accuracy = (correct / total) * 100 if total > 0 else 0.0
    incorrect = total - correct
    print(
        f"SupervisedMCTS Test Data Accuracy: {accuracy:.2f}% over {total} samples. {incorrect} incorrect."
    )

    return accuracy
