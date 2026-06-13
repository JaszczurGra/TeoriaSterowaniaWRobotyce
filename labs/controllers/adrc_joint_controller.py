import numpy as np
from observers.eso import ESO
from .controller import Controller


class ADRCJointController(Controller):
    def __init__(self, b, kp, kd, p, q0, Tp):
        self.b = b
        self.kp = kp
        self.kd = kd

        A = np.array([[0., 1., 0.],
                      [0., 0., 1.],
                      [0., 0., 0.]])
        B = np.array([[0.], [b], [0.]])
        W = np.array([[1., 0., 0.]])
        L = np.array([[3. * p], [3. * p ** 2], [p ** 3]])
        self.eso = ESO(A, B, W, L, q0, Tp)

    def set_b(self, b):
        self.b = b
        self.eso.set_B(np.array([[0.], [b], [0.]]))

    def calculate_control(self, x, q_d, q_d_dot, q_d_ddot):
        q = x[0]
        z = self.eso.get_state()
        q_dot_hat = z[1]
        f_hat = z[2]

        v = self.kp * (q_d - q) + self.kd * (q_d_dot - q_dot_hat) + q_d_ddot
        u = (v - f_hat) / self.b

        self.eso.update(q, u)
        return u
