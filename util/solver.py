import numpy as np
from games.connect4 import Connect4


class Solver:
    def __init__(self, game: Connect4):
        self.game: Connect4 = game
        self.depth_limit = 20  # we'll need a openning moves dataset for larger depths

    def evaluate_state(self):
        policy = np.zeros(self.game.num_of_cols)
        # fastest win is 7 moves
        best_score = self.game.num_of_cols * self.game.num_of_rows - 7
        worst_score = (self.game.num_of_cols * self.game.num_of_rows - 7) * -1
        value = worst_score  # worst case is the fastest loss

        # separate handling for valid moves. invalid moves always have a 0 probability
        moves_values = {}

        # NOTE: assumes player = 1
        best_eval = float("-inf")
        for move in self.game.get_legal_moves():
            self.game.make_move(move)
            value = self.min_value(0, float("-inf"), float("inf"))
            self.game.undo_move()
            moves_values[move] = value
            if value > best_eval:
                best_eval = value
                best_move = move

        # normalize the moves values and convert to value-weighted probabilities
        max_value = max(moves_values.values())
        min_value = min(moves_values.values())
        if max_value != min_value:
            for move, value in moves_values.items():
                moves_values[move] = (value - min_value) / (max_value - min_value)
            total = sum(moves_values.values())
            for move, value in moves_values.items():
                policy[move] = value / total
        else:
            # equal probability for all moves
            for move in moves_values.keys():
                policy[move] = 1 / len(moves_values)

        # value of state = value of best move, scaled between [0, 1] aka between worst and best
        value = (best_eval - worst_score) / (best_score - worst_score)
        # clamping to [-1, 1]
        value = 2 * value - 1

        return policy, value

    def max_value(self, depth, alpha, beta):
        """
        Maximizing layer of the minimax algorithm with alpha-beta pruning.
        """
        result = self.game.evaluate_board()
        if depth == self.depth_limit:
            return 0  # we assume everything is a tie
        elif result is not None:
            return result

        max_eval = float("-inf")
        for move in self.game.get_legal_moves():
            # if depth == 0:
            #     print(f"Move {move}, Alpha: {alpha}, Beta: {beta}")
            self.game.make_move(move)
            eval = self.min_value(depth + 1, alpha, beta)
            self.game.undo_move()
            max_eval = max(max_eval, eval)
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        # if self.debug and depth <= self.debug_depth:
        # if depth <= self.debug_depth:
        #     print(" " * depth + f"Max-Value: {max_eval} For depth {depth} and move {move}")
        return max_eval

    def min_value(self, depth, alpha, beta):
        """
        Minimizing layer of the minimax algorithm with alpha-beta pruning.
        """
        result = self.game.evaluate_board()
        if depth == self.depth_limit:
            return 0  # we assume everything is a tie
        elif result is not None:
            return result

        min_eval = float("inf")
        for move in self.game.get_legal_moves():
            self.game.make_move(move)
            eval = self.max_value(depth + 1, alpha, beta)
            # if depth == 1:
            #     print(f"Sub-move: {move} with eval {eval}")
            self.game.undo_move()
            min_eval = min(min_eval, eval)
            beta = min(beta, eval)
            if beta <= alpha:
                break
        # if self.debug and depth <= self.debug_depth:
        # if depth <= self.debug_depth:
        # print(f"Min-Value: {min_eval} For depth {depth} and move {move}")
        # print(" " * depth + f"Min-Value: {min_eval} For depth {depth} and move {move}")
        return min_eval
