import numpy as np
from models.manipulator_model import ManiuplatorModel
from .controller import Controller


class FeedbackLinearizationController(Controller):
    def __init__(self, Tp):
        self.model = ManiuplatorModel(Tp)
        # PD gains for the linearized double-integrator error dynamics
        # (e_ddot + Kd*e_dot + Kp*e = 0). Critically damped: Kd = 2*sqrt(Kp).
        self.Kp = np.diag([25., 25.])
        self.Kd = np.diag([10., 10.])

    def calculate_control(self, x, q_r, q_r_dot, q_r_ddot):
        """
        Feedback linearization (eq. 25) with an outer PD loop (eq. 27):
            v = q_r_ddot + Kd*(q_r_dot - q_dot) + Kp*(q_r - q)
            u = M(x) * v + C(x) * q_dot
        """
        q = x[:2]
        q_dot = x[2:]

        v = q_r_ddot + self.Kd @ (q_r_dot - q_dot) + self.Kp @ (q_r - q)
        u = self.model.M(x) @ v + self.model.C(x) @ q_dot
        return u
