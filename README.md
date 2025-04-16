# Connect 4 - AlphaZero

The AlphaZero algorithm is a modified **Monte-Carlo Tree Search (MCTS)** backed by a deep residual neural network (or "**ResNet**").
AlphaZero plays against itself, with each side choosing moves selected by MCTS.  

## The Neural Network
Input: the board state, a 6x7 matrix of 0s, 1s, and -1s.
Ouput: the value of the position ranging from 1 (win) to -1 (loss) and a vector of prior probabilities for each possible move.

## MCTS
1. Tree traversal
2. Node expansion
3. Rollout (random simulation, maybe we could do a-b pruning instead?)
4. Backpropagation

## AlphaZero 
Rollouts are replaced by fetching predictions from the NN, and UCB1 is replaced by PUCT (polynomial upper confidence tree). The algorithm looks like this:

## Resources
[Deep Residual Learning for Image Recognition (ResNet)](https://arxiv.org/abs/1512.03385)  
[Lessons from AlphaZero: Connect Four](https://medium.com/oracledevs/lessons-from-alphazero-connect-four-e4a0ae82af68)  




## How to Generate Training Data
Purpose: The generate.py script generates labeled training data for Connect4 using a solver. The data includes board states, policies, and values.

Run the Script: `python -m util.generate`

Output:

The generated training data is saved as a .npy file in the data/ directory.
Default file path: `data/connect4_4x4_training_data.npy`.
Notes:

The script generates 10,000 samples by default.
If the file already exists, it will prompt before overwriting.


## How to Train the Model
Purpose: The connect4_train.py script trains a neural network on the generated training data.

Run the Script: `python -m util.connect4_train`

Output:

The trained model is saved as a .pt file in the models/ directory.
Default file path: `models/connect4_4x4_supervised.pt`.
Notes:

Ensure the training data file (`data/connect4_4x4_training_data.npy`) exists before running the script.
The script splits the data into training (80%) and testing (20%) sets and trains the model for 10 epochs by default.

## AlphaZero - Self Play
Using the the generated training data games, we run mcts with an randomly initialized neural network.  
In batches, lets say of 100, we can stop self-play and train the neural network with the results of those games.  


The MuZero implementation (a successor to AlphaZero) reportedly uses a fixed-size replay buffer that continuously replaces old data, effectively implementing a sliding window approach that "forgets" early training data.  


What we really care about is how good the generated training data is. We can run this and see how it does.  