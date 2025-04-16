import os
import random
import numpy as np
from tqdm import tqdm
from util.solver import Solver
from games.connect4 import Connect4

# generating labelled training data, we feed random board positions to a Connect Four solver
# generate our labelled sets using positions from highly imperfect games (epsilon-greedy policy)

game = Connect4(num_of_rows=4, num_of_cols=4)
solver = Solver(game)

# x = flattened board
# y = policy, value
training_data = []
unique_states = set()


def hash_board(state, value):
    # Map board values (-1, 0, 1) to (2, 0, 1) for compact representation
    mapped_state = (np.array(state) + 1).flatten()  # Flatten and map values
    state_str = "".join(map(str, mapped_state))  # Convert to a single string
    return hash((state_str, value))


def decode_board(hash_key):
    # Decode the hash key back to board state and value
    state_str, value = hash_key
    # Convert the string back to a flattened array
    mapped_state = np.array(list(map(int, state_str)))
    # Reverse the mapping (2 -> -1, 0 -> 0, 1 -> 1)
    state = (mapped_state - 1).reshape(game.num_of_rows, game.num_of_cols)
    return state, value


def flip_board(state):
    # Flip the board horizontally
    return np.array([row[::-1] for row in state])


num_samples = 50_000
# num_samples = 100

policy, value = solver.evaluate_state()
move_count = 0
with tqdm(total=num_samples, desc="Generating samples") as pbar:
    while len(training_data) < num_samples:
        move_count += 1
        # genration policy: 80% chance to move random
        if random.random() < 0.80:
            legal_moves = game.get_legal_moves()
            action = random.choice(legal_moves)
        else:
            best_move = np.argmax(policy)
            action = best_move

        game.make_move(action)
        game.evaluate_board()

        # game.print_pretty()

        if game.result is not None:
            game.reset()
            move_count = 0

        if move_count <= 3:  # save some time, for 4x4 first 3 moves, all ties.
            policy = [0.25, 0.25, 0.25, 0.25]
            value = 0.0
        else:
            policy, value = solver.evaluate_state()

        # add entry to training_data if the game state is unique
        hash_key = hash_board(game.board, value)
        if hash_key not in unique_states:
            unique_states.add(hash_key)
            training_data.append((game.board.copy(), policy.copy(), value))
            pbar.update(1)
        # add mirror to training_data
        flipped_board = flip_board(game.board)
        hash_key_flipped = hash_board(flipped_board, value)
        if hash_key_flipped not in unique_states:
            unique_states.add(hash_key_flipped)
            flipped_policy = policy[::-1]
            training_data.append((flipped_board.copy(), flipped_policy, value))
            pbar.update(1)

# Summary
print("Generated training data:")
print(f"Total unique states generated: {len(unique_states)}")
print(f"Total training samples: {len(training_data)}")


print("10 Examples From Sample Data:")
for i in range(10):
    print("Sample", i + 1)
    sample = random.choice(training_data)
    state, action, value = sample
    print("Sample State (Board):")
    sample_game = Connect4(num_of_rows=4, num_of_cols=4, board=state)
    sample_game.print_pretty()
    print("Sample Policy:")
    print(np.array(action))
    print("Sample Value:")
    print(value)


file_name = "data/connect4_4x4_training_data_50k.npy"

# are you sure.
file_exists = os.path.isfile(file_name)
if file_exists:
    print(f"File {file_name} already exists. It will be overwritten.")
input("Press Enter to save the training data to a .npy file...")

# Save the training data to a .npy file
training_data_array = np.array(training_data, dtype=object)
np.save(file_name, training_data_array)
print(f"Training data saved to {file_name}")
