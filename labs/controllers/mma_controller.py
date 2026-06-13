import numpy as np
from models.manipulator_model import ManiuplatorModel
from .controller import Controller


class MMAController(Controller):
    def __init__(self, Tp):
        self.Tp = Tp
        self.models = [
            ManiuplatorModel(Tp, m3=0.1, r3=0.05),
            ManiuplatorModel(Tp, m3=0.01, r3=0.01),
            ManiuplatorModel(Tp, m3=1.0, r3=0.3),
        ]
        self.i = 0

        self.Kp = np.diag([25., 25.])
        self.Kd = np.diag([10., 10.])

        self.x_prev = None
        self.u_prev = None

    def choose_model(self, x):
        if self.x_prev is None or self.u_prev is None:
            return
        errors = []
        for model in self.models:
            x_pred = self._predict(model, self.x_prev, self.u_prev)
            errors.append(np.linalg.norm(x - x_pred))
        self.i = int(np.argmin(errors))

    def _predict(self, model, x, u):
        q_dot = x[2:]
        q_ddot = np.linalg.inv(model.M(x)) @ (u - model.C(x) @ q_dot)
        x_dot = np.concatenate([q_dot, q_ddot])
        return x + self.Tp * x_dot

    def calculate_control(self, x, q_r, q_r_dot, q_r_ddot):
        self.choose_model(x)
        q = x[:2]
        q_dot = x[2:]

        v = q_r_ddot + self.Kd @ (q_r_dot - q_dot) + self.Kp @ (q_r - q)
        M = self.models[self.i].M(x)
        C = self.models[self.i].C(x)
        u = M @ v + C @ q_dot

        self.x_prev = np.array(x, dtype=float)
        self.u_prev = u
        return u
