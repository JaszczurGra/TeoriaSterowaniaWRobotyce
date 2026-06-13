from copy import copy
import numpy as np


class ESO:

    def __init__(self, A, B, W, L, state, Tp):
        self.A = A
        self.B = B
        self.W = W
        self.L = L
        self.state = np.pad(np.array(state, dtype=float), (0, A.shape[0] - len(state)))
        self.Tp = Tp
        self.states = []

    def set_B(self, B):
        self.B = B

    def update(self, q, u):
        self.states.append(copy(self.state))

        q = np.reshape(q, -1)
        u = np.reshape(u, -1)

        z_dot = self.A @ self.state + self.B @ u + self.L @ (q - self.W @ self.state)
        self.state = self.state + self.Tp * z_dot

    def get_state(self):
        return self.state
