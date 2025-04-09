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
        self.total_score = 0 # cumulative reward (from the root's perspective)
        self.untried_moves = game.get_legal_moves()  # moves that have not yet been expanded
        if self.parent is not None:
            self.turn = -self.parent.turn
        else:
            self.turn = 1

    def is_fully_expanded(self):
        return len(self.untried_moves) == 0

    def best_child(self, exploration_constant=1.41, debug=False):
        """
        self explanatory, select the child with the highest UCT value.
        """
        best_score = -float("inf")
        best_child = None
        for move, child in self.children.items():
            if child.num_visits == 0:
                return child # TODO: choose random child instead of first
            score = (child.total_score / child.num_visits) + exploration_constant * math.sqrt(math.log(self.
            num_visits) / child.num_visits)
            if debug:
                print(f"Move: {move}, Score: {score}, Visits: {child.num_visits}, Total Score: {child.total_score}")
            if score > best_score:
                best_score = score
                best_child = child
        return best_child
    
    def PUCT(self, exploration_constant=1.41, policy_priors=None, debug=False):
        """
        Select the child with the highest PUCT value.
        
        PUCT formula: Q(s,a) + c_puct * P(s,a) * sqrt(sum_b N(s,b)) / (1 + N(s,a))
        """
        best_score = -float("inf")
        best_child = None
        
        # Calculate total visits for the square root term
        total_visits = sum(child.num_visits for child in self.children.values())
        sqrt_total_visits = math.sqrt(total_visits) if total_visits > 0 else 1.0
        
        for move, child in self.children.items():
            # If child has no visits, prioritize it
            if child.num_visits == 0:
                return child
            
            # Exploitation term (Q-value)
            q_value = child.total_score / child.num_visits
            
            # Prior probability for this move
            prior = policy_priors[move] if policy_priors is not None else 1.0 / len(self.children)
            
            # PUCT exploration bonus
            u_value = exploration_constant * prior * sqrt_total_visits / (1 + child.num_visits)
            
            # Combined PUCT score
            puct_score = -(q_value + u_value)
            
            if debug:
                print(f"Move: {move}, PUCT: {puct_score:.4f}, Q: {q_value:.4f}, U: {u_value:.4f}, "
                  f"Visits: {child.num_visits}, Prior: {prior:.4f}")
        
        if puct_score > best_score:
            best_score = puct_score
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

        self.root = None

    def evaluate_with_model(self, game: Connect4) -> float:
        """
        evaluate the current game state using the trained model.
        returns the value prediction (expected outcome) as a float in [-1, 1].
        """
        board = game.board
        board_tensor = convert_board_to_tensor(board).unsqueeze(0)  # add batch dimension
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

    def rollout(self, game: Connect4, root_player: int = 1) -> float:
        """
        using the trained model to evaluate the leaf node.
        if the game is terminal, compute a scaled reward based on move count (as before).
        otherwise, return the network's value prediction.
        """
        result = game.evaluate_board()
        if result is not None:
            best_score = game.num_of_rows * game.num_of_cols - 7
            worst_case = best_score * -1
            # print("best_score", best_score, "worst_case", worst_case)
            value = (result - worst_case) / (best_score - worst_case)
            value = 2 * value - 1  # scale to [-1, 1]
            return value



            # worst_case = game.num_of_rows * game.num_of_cols
            # if result > 0:
            #     winner = 1
            # elif result < 0:
            #     winner = -1
            # else:
            #     winner = 0

            # scaled = (worst_case - game.move_count) / worst_case
            # if winner == root_player:
            #     return scaled
            # elif winner == 0:
            #     return 0.0
            # else:
            #     return -scaled

        # if non-terminal state then use the model's prediction.
        return self.evaluate_with_model(game)

    def search(self, game: Connect4) -> np.ndarray:
        """
        Run MCTS simulations from the given game state using the trained model at the leaf nodes.
        Returns a probability vector over moves (for every column) derived from the visit counts at the root.
        """
        root_player = 1
        root_node = MCTSNode(clone_game(game))
        self.root = root_node

        for _ in range(self.iterations):
            node = root_node
                    
            simulation_game = clone_game(game)

            # selection process
            # while node.is_fully_expanded() and node.children:
            while len(node.children) > 0 and node.game.result is None:
                # node = node.best_child(self.exploration_constant)
                policy_priors = self.evaluate_policy_with_model(simulation_game)
                # PUCT
                node = node.PUCT(
                    exploration_constant=self.exploration_constant,
                    policy_priors=policy_priors,
                )
                simulation_game.make_move(node.move)

            # expansion process
            # if num_visits is zero, we expand this node
            if node.num_visits == 0:
                for move in node.untried_moves:
                    simulation_game.make_move(move)
                    child_node = MCTSNode(clone_game(simulation_game), parent=node, move=move)
                    node.children[move] = child_node
                    simulation_game.undo_move()
                # current is a random untried move
                move = random.choice(node.untried_moves)
                simulation_game.make_move(move)
                node.untried_moves.remove(move)
                node = node.children[move]

            # #save the board state from where the evaluation will start.
            # sample_state = np.copy(simulation_game.board)

            #evaluation using the Model instead of rollout as we discussed.
            reward = self.rollout(clone_game(simulation_game), root_player)

            # #save the sample
            # self.samples.append((sample_state, reward))

            # backpropogation
            # print("backpropogation reward:", reward * node.turn)
            back_propped_move = node.move
            while node is not None:
                current_reward = reward * node.turn
                node.num_visits += 1
                node.total_score += current_reward
                # node.game.print_pretty()
                # print("back_propped_move", back_propped_move, "is parent: ", node.parent is None)
                # print("reward", current_reward)

                # current_reward = -current_reward  # TODO: review flip reward for the opponent

                node = node.parent

        #compute move probabilities for the root node using normalized visit counts.
        move_visits = {move: child.num_visits for move, child in root_node.children.items()}
        total_visits = sum(move_visits.values())
        move_probs = np.zeros(game.num_of_cols)
        for move, num_visits in move_visits.items():
            move_probs[move] = num_visits / total_visits
        return move_probs

    def predict(self, game: Connect4):
        """
        Evaluate the current game state using the trained model.
        Returns:
            - policy: A probability distribution over moves (numpy array).
            - value: A scalar value prediction in [-1, 1].
        """
        board_tensor = convert_board_to_tensor(game.board).unsqueeze(0)  # add batch dimension
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


# evaluate_supervised_mcts_on_test_data(
#     num_samples=50, mcts_iterations=800
# )
