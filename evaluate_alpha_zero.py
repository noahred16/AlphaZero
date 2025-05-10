# np
import numpy as np


data_files = [
    "connect4_4x4_self_play_data_50k.npy",
    "connect4_4x4_training_data.npy",
    "connect4_4x4_training_data_100k.npy",
    "connect4_4x4_training_data_50k.npy",
]

csv_files = [
    "supervised_mcts_evaluation.csv",
    "supervised_mcts_evaluation_100k.csv",
]


# preview first row of each data file
for data_file in data_files:
    data_path = f"data/{data_file}"
    training_data = np.load(data_path, allow_pickle=True)
    print(f"Data file: {data_file}")
    print("Shape of data:", training_data.shape)
    print("First row of data:", training_data[0].shape)
    print("First sample board:")
    print(training_data[0])
    print("-" * 40)
