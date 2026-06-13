import numpy as np


class ManiuplatorModel:
    def __init__(self, Tp, m3=0.0, r3=0.01):
        self.Tp = Tp
        self.l1 = 0.5
        self.r1 = 0.01
        self.m1 = 1.
        self.l2 = 0.5
        self.r2 = 0.01
        self.m2 = 1.
        self.I_1 = 1 / 12 * self.m1 * (3 * self.r1 ** 2 + self.l1 ** 2)
        self.I_2 = 1 / 12 * self.m2 * (3 * self.r2 ** 2 + self.l2 ** 2)
        self.m3 = m3
        self.r3 = r3
        self.I_3 = 2. / 5 * self.m3 * self.r3 ** 2

    def M(self, x):
        q1, q2, q1_dot, q2_dot = x
        d1 = self.l1 / 2
        d2 = self.l2 / 2
        alpha = self.I_1 + self.I_2 + self.m1 * d1 ** 2 + self.m2 * (self.l1 ** 2 + d2 ** 2) + \
                self.I_3 + self.m3 * (self.l1 ** 2 + self.l2 ** 2)
        beta = self.m2 * self.l1 * d2 + self.m3 * self.l1 * self.l2
        delta = self.I_2 + self.m2 * d2 ** 2 + self.I_3 + self.m3 * self.l2 ** 2
        m_11 = alpha + 2 * beta * np.cos(q2)
        m_12 = delta + beta * np.cos(q2)
        m_21 = m_12
        m_22 = delta
        return np.array([[m_11, m_12], [m_21, m_22]])

    def C(self, x):
        q1, q2, q1_dot, q2_dot = x
        d2 = self.l2 / 2
        beta = self.m2 * self.l1 * d2 + self.m3 * self.l1 * self.l2
        c_11 = -beta * np.sin(q2) * q2_dot
        c_12 = -beta * np.sin(q2) * (q1_dot + q2_dot)
        c_21 = beta * np.sin(q2) * q1_dot
        c_22 = 0
        return np.array([[c_11, c_12], [c_21, c_22]])
