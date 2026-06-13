import numpy as np

from models.free_model import FreeModel
from observers.eso import ESO
from .controller import Controller
from models.ideal_model import IdealModel


class ADRFLController(Controller):
    def __init__(self, Tp, q0, Kp, Kd, p):
        self.Tp = Tp
        self.model = IdealModel(Tp)
        self.free_model = FreeModel(Tp)
        self.Kp = Kp
        self.Kd = Kd

        p = np.asarray(p, dtype=float)
        self.L = np.vstack([3. * np.diag(p), 3. * np.diag(p ** 2), np.diag(p ** 3)])

        W = np.hstack([np.eye(2), np.zeros((2, 4))])
        A, B = self.free_model.state_space(q0[:2], q0[2:])
        self.eso = ESO(A, B, W, self.L, q0, Tp)
        self.update_params(q0[:2], q0[2:])

    def update_params(self, q, q_dot):
        A, B = self.free_model.state_space(q, q_dot)
        self.eso.A = A
        self.eso.B = B

    def calculate_control(self, x, q_d, q_d_dot, q_d_ddot):
        q = np.asarray(x[:2], dtype=float)
        z = self.eso.get_state()
        q_dot_hat = z[2:4]
        f_hat = z[4:6]

        self.update_params(q, q_dot_hat)

        v = self.Kp @ (q_d - q) + self.Kd @ (q_d_dot - q_dot_hat) + q_d_ddot

        xq = np.array([q[0], q[1], q_dot_hat[0], q_dot_hat[1]])
        M = self.model.M(xq)
        C = self.model.C(xq)
        u = M @ (v - f_hat) + C @ q_dot_hat

        self.eso.update(q, u)
        return u
