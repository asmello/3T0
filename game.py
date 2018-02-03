import re
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
        return State.domain[np.argmax(self.mcts.search(eps=0))]

    @property
    def winner(self):
        return State.player_codes[self.mcts.state.winner]

    def apply(self, action):
        if action not in self.mcts.state.actions:
            raise IllegalActionException(f"tried illegal action {action}")
        self.mcts.apply(action)
        return self

    @staticmethod
    def coord_to_action(coord):
        m = re.search("([a-zA-Z])(\d)$", coord)
        if m:
            return State.raw_shape[0] * (ord(m[1].upper()) - ord('A')) \
                    + int(m[2]) - 1

    @staticmethod
    def action_to_coord(action):
        return chr(action // 9 + ord('A')) + str(action % 9 + 1)

    def __repr__(self):
        return str(self.mcts.state)


class IllegalActionException(Exception):
    pass
