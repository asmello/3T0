# U3T0
A minimal AlphaGoZero-like AI that plays Ultimate Tic Tac Toe.

# Features
## Implemented
- MCTS modified algorithm
- Temperature control
  - Tau set to 0.1 at cutoff
  - Cutoff default at 20 moves
- Dirichlet noise at root (same setting as AGZ)
- Dihedral symmetry compensation
- Model selection (switch at 5% advantage)
  - Evaluation every 80 episodes (default)
  - 20 games used for evaluation (default)
- History truncation (default at -5 updates)
- L2 loss (set to 1e-4 by default)

## Missing
- Resignation (unusual for UTTT)
- Asynchronous training
- Batch evaluation

## Extra
- Shortcut evaluation for winning state
- Very simple human playing interface
- Command-line selection of some parameters
- Model persistence at every switch

# About
This is meant to be a concise and didactic implementation of the AGZ AI model and its neural net architecture. It's not very fast, mostly because of lack of asynchronous training, batch evaluation and distributed computation. However, it should run on any decent personal computer.

By modifying the `state.py` implementation, it should be possible to make the AI learn any similar enough board game. The assumptions made are the following:

- The game has two players (1 and -1) that alternate their moves;
- The entire state of the game can be represented by a stack of 2D matrices;
- There are finite actions to choose from at each state;
- The actions can be encoded by a fixed dense domain ![domain](http://latex.codecogs.com/svg.latex?%5B0%2C%20k%5D%20%5Cin%20%5Cmathbb%7BZ%7D%5E&plus;)
