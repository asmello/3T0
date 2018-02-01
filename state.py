import numpy as np


class State:
    '''UltimateTicTacToe game state.'''

    domain = np.arange(81)
    raw_shape = (9, 9, 3)
    player_codes = {1: 'X', 0: '-', -1: 'O'}

    def __init__(self, board=None, cell=None, player=1):
        '''Note: if board specified, cell is required.'''

        self.player = player
        self.full = False
        self.over = False
        self.winner = 0
        self.cell_winner = np.zeros((3, 3))
        self.cell_full = np.zeros((3, 3))
        self.active_cell = cell

        # no board --> initial state
        if board is None:
            self.board = np.zeros((9, 9))
            self.actions = State.domain.copy()
            active_squares = np.ones((9, 9))

        else:
            self.board = board
            active_squares = np.zeros((9, 9))

            fields = [
                [0, 1, 2], [3, 4, 5], [6, 7, 8],  # rows
                [0, 3, 6], [1, 4, 7], [2, 5, 8],  # columns
                [0, 4, 8], [2, 4, 6]              # diagonals
            ]

            # figure out each cell's status
            for i in range(3):
                for j in range(3):
                    cell = self.board[3*i:3*i+3, 3*j:3*j+3].flatten()
                    self.cell_full[i, j] = np.all(cell)
                    for idx in fields:
                        t = cell[idx]
                        a = np.min(t)
                        b = np.max(t)
                        if a and a == b:
                            self.cell_winner[i, j] = a
                            break

            self.cell_locked = np.logical_or(self.cell_winner, self.cell_full)

            # figure out global board status
            self.full = np.all(self.cell_locked)
            winners = self.cell_winner.flatten()
            for idx in fields:
                t = winners[idx]
                a = np.min(t)
                b = np.max(t)
                if a and a == b:
                    self.winner = a
                    break

            self.over = self.winner or self.full

            # figure out the legal moves
            self.actions = []
            if not self.over:
                i, j = self.active_cell
                if self.cell_locked[i, j]:
                    # can pick any unlocked cell
                    for i in range(3):
                        for j in range(3):
                            if self.cell_locked[i, j]:
                                continue
                            # any empty square within the cell
                            for u in range(3*i, 3*i+3):
                                for v in range(3*j, 3*j+3):
                                    active_squares[u, v] = 1
                                    if not self.board[u, v]:
                                        self.actions.append(9 * u + v)
                else:
                    # any empty square within the cell
                    for u in range(3*i, 3*i+3):
                        for v in range(3*j, 3*j+3):
                            active_squares[u, v] = 1
                            if not self.board[u, v]:
                                self.actions.append(9 * u + v)

        # this works as input for the Estimator class
        self.raw = np.stack((self.board, active_squares,
                            np.full((9, 9), self.player)),
                            axis=-1)

    def apply(self, action):
        board = self.board.copy()
        i, j = action // 9, action % 9
        board[i, j] = self.player
        return State(board, (i % 3, j % 3), -self.player)

    def __repr__(self):
        s = '\n'
        for i in range(9):
            cells = ' '.join(State.player_codes[x] for x in self.board[i, :])
            s += ' ' * 4 + cells + '\n'
        return s
