import numpy as np

from .manipulator_model import ManiuplatorModel


class FreeModel(ManiuplatorModel):
    def __init__(self, Tp, m3=0.0, r3=0.01):
        super().__init__(Tp, m3, r3)

    def state_space(self, q, q_dot):
        x = np.array([q[0], q[1], q_dot[0], q_dot[1]])
        M = self.M(x)
        C = self.C(x)
        M_inv = np.linalg.inv(M)
        M_inv_C = M_inv @ C

        Z = np.zeros((2, 2))
        I = np.eye(2)
        A = np.block([[Z,         I,        Z],
                      [Z,        -M_inv_C,  I],
                      [Z,         Z,        Z]])
        B = np.block([[Z],
                      [M_inv],
                      [Z]])
        return A, B
