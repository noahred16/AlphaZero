import random
import math
import numpy as np
import copy
from games.connect4 import Connect4

def clone_game(game: Connect4) -> Connect4:
    """
    creates a copy of the Connect4 game state. 
    done to make sure simulations does not affect original board state. 
    """
    new_board = np.copy(game.board)
    new_game = Connect4(
        board=new_board, 
        current_player=game.current_player, 
        num_of_rows=game.num_of_rows, 
        num_of_cols=game.num_of_cols
    )
    new_game.move_count = game.move_count
    new_game.move_history = game.move_history.copy()
    new_game.result = game.result
    return new_game


class MCTSNode:
    def __init__(self, game: Connect4, parent=None, move=None): # boiler plate stuff for initialization
        self.game = game  # game state at this node (a clone of Connect4)
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
        select the child with the highest UCT (Upper Confidence Bound for Trees) value.
        """
        best_score = -float("inf")
        best_child = None
        for move, child in self.children.items():
            # UCT formula: (wins/visits) + c * sqrt(ln(parent.visits)/child.visits)
            score = (child.wins / child.visits) + exploration_constant * math.sqrt(math.log(self.visits) / child.visits)
            if score > best_score:
                best_score = score
                best_child = child
        return best_child


class MCTS:
    def __init__(self, iterations=1000, exploration_constant=1.41):
        self.iterations = iterations
        self.exploration_constant = exploration_constant
        self.samples = []  # will store tuples of (board state, outcome)

    # def rollout(self, game: Connect4, root_player: int) -> int:
    #     """
    #     Play out the game randomly until reaching a terminal state.
    #     Returns:
    #         1  if the terminal state is a win for the root player,
    #         -1 if it is a loss,
    #         0  for a tie.
    #     """
    #     while True:
    #         result = game.evaluate_board()
    #         if result is not None:
    #             # determine the winner: positive means player 1 wins,
    #             # negative means player -1 wins, 0 indicates a tie.
    #             if result > 0:
    #                 winner = 1
    #             elif result < 0:
    #                 winner = -1
    #             else:
    #                 winner = 0

    #             if winner == root_player:
    #                 return 1
    #             elif winner == 0:
    #                 return 0
    #             else:
    #                 return -1

    #         legal_moves = game.get_legal_moves()
    #         if not legal_moves:
    #             return 0  # tie if no moves available
    #         move = random.choice(legal_moves)
    #         game.make_move(move)
    
    def rollout(self, game: Connect4, root_player: int) -> float:
        """
        Play out the game randomly until reaching a terminal state.
        Returns a reward that factors in the number of moves taken:
        - A quicker win yields a higher positive reward.
        - A slower win yields a lower positive reward.
        - For a loss, the reward is negative, with faster losses more severely penalized.
        - Ties return 0.
        """
        while True:
            result = game.evaluate_board()
            if result is not None:
                worst_case = game.num_of_rows * game.num_of_cols  # maximum possible moves
                # Determine the winner based on the last move.
                if result > 0:
                    winner = 1
                elif result < 0:
                    winner = -1
                else:
                    winner = 0

                # Calculate a scaling factor based on how quickly the game ended.
                # For example, a win achieved in fewer moves yields a larger factor.
                scaled = (worst_case - game.move_count) / worst_case

                if winner == root_player:
                    return scaled
                elif winner == 0:
                    return 0.0
                else:
                    return -scaled

            legal_moves = game.get_legal_moves()
            if not legal_moves:
                return 0.0  # Tie if no moves are available.
            move = random.choice(legal_moves)
            game.make_move(move)

    def search(self, game: Connect4) -> np.ndarray:
        """
        Run MCTS simulations from the given game state.
        Returns a probability vector over moves for every column,
        computed from the visit counts at the root.
        """
        root_player = game.current_player
        root_node = MCTSNode(clone_game(game))

        for _ in range(self.iterations):
            node = root_node
            simulation_game = clone_game(game)

            # selection process where we 
            # traverse the tree by selecting the best child until reaching a node
            # that is not fully expanded or is terminal.
            while node.is_fully_expanded() and node.children:
                node = node.best_child(self.exploration_constant)
                simulation_game.make_move(node.move)

            # expansion process:
            # if the node has untried moves, pick one and expand the tree.
            if node.untried_moves:
                move = random.choice(node.untried_moves)
                simulation_game.make_move(move)
                node.untried_moves.remove(move)
                child_node = MCTSNode(clone_game(simulation_game), parent=node, move=move)
                node.children[move] = child_node
                node = child_node

            #saving the board state from where the rollout will start.
            sample_state = np.copy(simulation_game.board)

            # running the rollout on this
            reward = self.rollout(clone_game(simulation_game), root_player)

            # storing this sample rn, will need for NN later ?
            self.samples.append((sample_state, reward))

            # backpropagation process:
            # propagate the simulation result up the tree, switching perspective at each level.
            current_reward = reward
            while node is not None:
                node.visits += 1
                node.wins += current_reward
                current_reward = -current_reward  # flip reward for the opponent
                node = node.parent

        # this computes move probabilities for the root node using normalized visit counts.
        move_visits = {move: child.visits for move, child in root_node.children.items()}
        total_visits = sum(move_visits.values())
        move_probs = np.zeros(game.num_of_cols)
        for move, visits in move_visits.items():
            move_probs[move] = visits / total_visits
        return move_probs

    def get_samples(self):
        """
        retrieve the board samples and outcomes collected during MCTS search.
        """
        return self.samples
