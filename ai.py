from mcts import MCTS
from state import State
from estimator import Estimator

import os.path
import numpy as np
from math import ceil


class AI:

    def __init__(self, load=None, filepath='best_estimator.h5'):
        self.filepath = filepath
        to_load = load or filepath
        if os.path.isfile(to_load):
            self.estimator = Estimator(State.raw_shape,
                                       len(State.domain),
                                       filepath=to_load)
        else:
            self.estimator = Estimator(State.raw_shape, len(State.domain))

    def duel(self, opponent, first=1):
        '''Play a full game against an opponent AI.'''

        if first == -1:
            e0, e1 = opponent, self.estimator
        else:
            e0, e1 = self.estimator, opponent

        s0, s1 = MCTS(e0), MCTS(e1)

        while not s0.state.over:

            a = State.domain[np.argmax(s0.search())]

            s0.apply(a)
            s1.apply(a)

            if s0.state.over:
                break

            a = State.domain[np.argmax(s1.search())]

            s1.apply(a)
            s0.apply(a)

        return s0.state.winner

    def simulate(self, first=1, tau_cutoff=20):
        '''Simulate a full game by self-playing.'''

        mcts = MCTS(self.estimator, first=first)
        history = []
        tau = 1.0

        while not mcts.state.over:

            if len(history) == tau_cutoff:
                tau = 0.1

            policy = mcts.search(tau)
            history.append((mcts.state.raw, policy))

            a = np.random.choice(State.domain, p=policy)
            mcts.apply(a)

        return history, mcts.state.winner

    def train(self, episodes=400, update_freq=80, eval_episodes=20):

        games = []

        for i in range(episodes):

            history, winner = self.simulate(first=np.random.choice([-1, 1]))
            print("Game --> winner:", State.player_codes[winner],
                  "moves:", len(history))
            games.append((history, winner))

            if i % update_freq + 1 == update_freq:

                print("Training new model...")
                new_estimator = self.estimator.update(games)

                score = 0
                for j in range(eval_episodes):
                    first = np.random.choice([-1, 1])
                    winner = self.duel(new_estimator, first=first)
                    score -= first * winner

                print("New model score:", score)
                if score >= ceil(0.05 * eval_episodes):
                    self.estimator = new_estimator
                    self.estimator.save(self.filepath)
                    print("New model selected.")
                else:
                    print("New model rejected.")

                games = games[-5*eval_episodes:]  # crop history
