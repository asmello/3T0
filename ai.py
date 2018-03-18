from mcts import MCTS
from state import State
from estimator import Estimator

import os.path
import numpy as np
from math import ceil


class AI:

    def __init__(self, load=None, filepath='best_estimator.h5',
                 num_episodes=400, eval_episodes=20, update_freq=80,
                 mcts_iters=100, tau_cutoff=20, max_games=10):
        self.num_episodes = num_episodes
        self.eval_episodes = eval_episodes
        self.update_freq = update_freq
        self.mcts_iters = mcts_iters
        self.tau_cutoff = tau_cutoff
        self.max_games = max_games
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

        s0 = MCTS(e0, maxiter=self.mcts_iters)
        s1 = MCTS(e1, maxiter=self.mcts_iters)

        while not s0.state.over:

            a = State.domain[np.argmax(s0.search(use_symmetry=True))]

            s0.apply(a)
            s1.apply(a)

            if s0.state.over:
                break

            a = State.domain[np.argmax(s1.search(use_symmetry=True))]

            s1.apply(a)
            s0.apply(a)

        return s0.state.winner

    def simulate(self, first=1):
        '''Simulate a full game by self-playing.'''

        mcts = MCTS(estimator=self.estimator, epsilon=0.25,
                    maxiter=self.mcts_iters, first=first)
        history = []
        tau = 1.0

        while not mcts.state.over:

            if len(history) == self.tau_cutoff:
                tau = 0.1

            policy = mcts.search(tau, sentinel=-1, use_symmetry=True)
            history.append((mcts.state.raw, policy))

            policy[policy < 0] = 0
            a = np.random.choice(State.domain, p=policy / np.sum(policy))
            mcts.apply(a)

        return history, mcts.state.winner

    def train(self):

        games = []
        winner_map = {-1: 'current', 0: 'neither', 1: 'new'}

        for i in range(self.num_episodes):

            history, winner = self.simulate(first=np.random.choice([-1, 1]))
            print("Game --> winner:", State.player_codes[winner],
                  "moves:", len(history))
            games.append((history, winner))

            if i % self.update_freq + 1 == self.update_freq:

                print("Training new model...")
                new_estimator = self.estimator.update(games)

                score = 0
                print("Evaluating...")
                for j in range(self.eval_episodes):
                    first = np.random.choice([-1, 1])
                    winner = self.duel(new_estimator, first=first)
                    print("Duel --> winner:", winner_map[-first * winner])
                    score -= first * winner

                print("New model score:", score)
                if score >= ceil(0.05 * self.eval_episodes):
                    self.estimator = new_estimator
                    self.estimator.save(self.filepath)
                    print("New model selected.")
                else:
                    print("New model rejected.")

                games = games[-self.max_games:] # truncate history
