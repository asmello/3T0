from state import State

import numpy as np
from math import sqrt


class Edge:

    def __init__(self, start, action, prior):
        self.total_value = 0  # W
        self.prior = prior  # P
        self.visits = 0  # N
        self.value = 0  # Q
        self.action = action
        self.start = start
        self.end = Node(start.state.apply(action), pedge=self)

    def update(self, v):
        self.visits += 1
        self.total_value += v
        self.value = self.total_value / self.visits
        self.start.visits += 1


class Node:

    def __init__(self, state=None, pedge=None, player=1):
        self.state = state or State(player=player)
        self.pedge = pedge
        self.edges = []
        self.visits = 0

    @property
    def is_leaf(self):
        return not self.edges

    @property
    def is_root(self):
        return not self.pedge

    def expand(self, estimator):
        p, v = estimator.compute(self.state)
        self.edges = [Edge(self, action, p[action])
                      for action in self.state.actions]
        return v

    def edge(self, action):
        for edge in self.edges:
            if edge.action == action:
                return edge


class MCTS:

    def __init__(self, estimator, first=1):
        self.node = Node(player=first)
        self.estimator = estimator

    @property
    def state(self):
        return self.node.state

    def apply(self, action):
        if self.node.is_leaf:
            self.node.expand(self.estimator)
        self.node = self.node.edge(action).end

    def search(self, tau=1.0, maxiter=100, c=1.0):

        # V(s, a) = Q(s, a) + U(s, a)
        def V(edge):
            return edge.value + c * edge.prior \
                * sqrt(node.visits) / (1 + edge.visits)

        node = self.node

        for i in range(maxiter):

            # select
            while not node.is_leaf:
                node = max(node.edges, key=V).end

            # expand
            if not node.state.over:
                v = node.expand(self.estimator)
            else:
                v = node.state.winner != 0

            # backup
            m = 1
            while node != self.node:
                node.pedge.update(v * m)
                node = node.pedge.start
                m *= -1

        visits = []
        for action in State.domain:
            edge = self.node.edge(action)
            visits.append(edge.visits if edge else 0)

        visits = np.array(visits) ** (1/tau)
        return visits / np.sum(visits)  # policy probability distribution (pi)
