import numpy as np
from models.manipulator_model import ManiuplatorModel
from .adrc_joint_controller import ADRCJointController
from .controller import Controller


class ADRController(Controller):
    def __init__(self, Tp, params, model_b=False):
        self.joint_controllers = []
        for param in params:
            self.joint_controllers.append(ADRCJointController(*param, Tp))
        self.model_b = model_b
        self.model = ManiuplatorModel(Tp) if model_b else None

    def calculate_control(self, x, q_d, q_d_dot, q_d_ddot):
        if self.model_b:
            M_inv = np.linalg.inv(self.model.M(x))
            for i, controller in enumerate(self.joint_controllers):
                controller.set_b(M_inv[i, i])

        u = []
        for i, controller in enumerate(self.joint_controllers):
            u.append(controller.calculate_control([x[i], x[i + 2]], q_d[i], q_d_dot[i], q_d_ddot[i]))
        return np.array(u)
