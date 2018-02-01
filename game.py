import numpy as np

from mcts import MCTS
from state import State


class Game:

    def __init__(self, ai, first=1):
        self.mcts = MCTS(ai.estimator, first=first)

    @property
    def over(self):
        return self.mcts.state.over

    @property
    def best_action(self):
        return State.domain[np.argmax(self.mcts.search())]

    @property
    def winner(self):
        return State.player_codes[self.mcts.state.winner]

    def apply(self, action):
        if action not in self.mcts.state.actions:
            raise RuntimeError("Illegal action!")
        self.mcts.apply(action)
        return self

    def __repr__(self):
        return str(self.mcts.state)
