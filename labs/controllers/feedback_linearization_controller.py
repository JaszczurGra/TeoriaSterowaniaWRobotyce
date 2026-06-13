import numpy as np
from models.manipulator_model import ManiuplatorModel
from .controller import Controller


class FeedbackLinearizationController(Controller):
    def __init__(self, Tp):
        self.model = ManiuplatorModel(Tp)
        self.Kp = np.diag([25., 25.])
        self.Kd = np.diag([10., 10.])

    def calculate_control(self, x, q_r, q_r_dot, q_r_ddot):
        q = x[:2]
        q_dot = x[2:]

        v = q_r_ddot + self.Kd @ (q_r_dot - q_dot) + self.Kp @ (q_r - q)
        u = self.model.M(x) @ v + self.model.C(x) @ q_dot
        return u
