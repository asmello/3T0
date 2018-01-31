import numpy as np


class State:
    '''TicTacToe game state.'''

    domain = list(range(9))
    raw_shape = (3, 3, 2)
    player_codes = {1: 'X', 0: '-', -1: 'O'}

    def __init__(self, board=None, player=1):
        self.player = player
        self.winner = 0

        # no board --> initial state
        if board is None:
            self.board = np.zeros(9)
            self.full = False

        else:
            self.board = board

            # check if this is a winning state
            fields = [
                [0, 1, 2], [3, 4, 5], [6, 7, 8],  # rows
                [0, 3, 6], [1, 4, 7], [2, 5, 8],  # columns
                [0, 4, 8], [2, 4, 6]              # diagonals
            ]
            for idx in fields:
                a = np.min(self.board[idx])
                b = np.max(self.board[idx])
                if a and a == b:
                    self.winner = a

            self.full = np.all(self.board)

        self.over = self.winner or self.full

    @property
    def raw(self):
        return np.stack((self.board.reshape(3, 3),
                        np.full((3, 3), self.player)),
                        axis=-1)

    @property
    def actions(self):
        return [] if self.over \
                  else np.flatnonzero(self.board == 0)

    def apply(self, action):
        board = self.board.copy()
        board[action] = self.player
        return State(board, -self.player)
