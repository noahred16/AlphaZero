import numpy as np


class Connect4:
    def __init__(self, board=None, num_of_rows=6, num_of_cols=7):
        self.num_of_rows = num_of_rows
        self.num_of_cols = num_of_cols
        self.board = np.zeros((self.num_of_rows, self.num_of_cols), dtype=int)
        self.result = None
        self.move_count = 0
        self.move_history = []

        if board is not None:
            self.board = np.array(board, dtype=int)
            player_1_moves = []
            player_2_moves = []
            for col in range(self.num_of_cols):
                for row in reversed(range(self.num_of_rows)):
                    if self.board[row][col] == 1:
                        player_1_moves.append((row, col))
                    elif self.board[row][col] == -1:
                        player_2_moves.append((row, col))

            n_p1 = len(player_1_moves)
            n_p2 = len(player_2_moves)
            if not (n_p1 == n_p2 or abs(n_p1 - n_p2) == 1):
                raise ValueError(
                    f"Invalid state: p1 moves ({n_p1}) and p2 moves ({n_p2}) are not balanced."
                )
            for i in range(max(n_p1, n_p2)):
                if i < n_p1:
                    self.move_history.append(player_1_moves[i])
                if i < n_p2:
                    self.move_history.append(player_2_moves[i])

            # if player 1 has the last move, flip
            if n_p1 > n_p2:
                self.flip_board()

            self.move_count = n_p1 + n_p2
            self.evaluate_board()

    def reset(self):
        self.board = np.zeros((self.num_of_rows, self.num_of_cols), dtype=int)
        self.result = None
        self.move_count = 0
        self.move_history = []

    def get_legal_moves(self):
        return [i for i in range(self.num_of_cols) if self.board[0][i] == 0]

    def make_move(self, column):
        # drop into first available row in the column
        for row in reversed(range(self.num_of_rows)):
            if self.board[row][column] == 0:
                self.board[row][column] = 1
                self.move_count += 1
                self.move_history.append((row, column))  # save move to history
                self.flip_board()
                return
        raise ValueError("Invalid move")

    def undo_move(self):
        """Remove the last token placed on the board."""
        if self.move_history:
            last_row, last_column = self.move_history.pop()
            self.board[last_row][last_column] = 0
            self.move_count -= 1
            self.flip_board()
            self.result = None
        else:
            raise ValueError("No moves to undo")

    def flip_board(self):
        """
        Flip the board for the next player.
        This is done to visualize the board the same regardless of whose turn it is.
        """
        self.board = -self.board

    # result is not set unless we evaluate. thats fine.
    def evaluate_board(self, player=1):
        """
        Evaluate the board for the current player, using last move made and last player
        """
        if self.move_count == 0 or self.move_count < 7:
            return None
        row, column = self.move_history[-1]

        directions = [
            (0, 1),  # Horizontal right
            (1, 0),  # Vertical down
            (1, 1),  # Diagonal down-right
            (1, -1),  # Diagonal down-left
        ]

        worst_case_num_moves = self.num_of_rows * self.num_of_cols  # (6*7=42)

        # check if the last move made a winning move
        # only need to check in 4 directions from the last move
        for dr, dc in directions:
            count = 1  # only need to check win for prev player (always -1)
            count += self.count_in_direction(row, column, dr, dc, -1)
            # print(f"Checking direction ({dr}, {dc}): count = {count}")
            count += self.count_in_direction(row, column, -dr, -dc, -1)
            # print(f"Checking direction ({-dr}, {-dc}): count = {count}")
            if count >= 4:
                self.result = self.scale_result()
                return self.result  # this is key
        # if board is full or has no more moves
        if self.move_count == worst_case_num_moves:
            self.result = 0
            return 0
        return None

    # scale results between [0.1, 1] relative to the fastest win
    def scale_result(self):
        n = self.move_count
        # 7 moves is the fastest win for connect4
        fastest_move = 7
        # max moves is cols * rows
        max_moves = self.num_of_cols * self.num_of_rows
        # get move value relative to fastest move
        value = 1 - (n - fastest_move) / (max_moves - fastest_move)
        # clamp value between [0.1, 1]
        clamp = 0.1
        result = (1 - clamp) * value + clamp

        return result

    # helper function for evaluate_board
    def count_in_direction(self, row, column, dr, dc, player):
        count = 0
        r, c = row + dr, column + dc
        while (
            0 <= r < self.num_of_rows
            and 0 <= c < self.num_of_cols
            and self.board[r][c] == player
        ):
            # print("r: ", r, "c: ", c, "player: ", player, "board: ", self.board[r][c])
            count += 1
            r += dr
            c += dc
        return count

    def print_pretty(self):
        # use X for player 1 and O for player -1 and empty for 0
        print("\n")
        for row in range(self.num_of_rows):
            print("|", end="")
            for col in range(self.num_of_cols):
                if self.board[row][col] == 1:
                    print("X|", end="")
                elif self.board[row][col] == -1:
                    print("O|", end="")
                else:
                    print(" |", end="")
            print("")
        print("---------------")
        for i in range(self.num_of_cols):
            print(f" {i}", end="")
        print("\n")

    def print_url(self):
        # https://connect4.gamesolver.org/?pos=44444456233333565556621211
        url = "https://connect4.gamesolver.org/?pos="
        # loop through history
        for row, column in self.move_history:
            url += str(column + 1)
        print(url)

    def ugly_print(self):
        print(self.board)
