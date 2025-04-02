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






