"""Lab 3 - ADRC + feedback linearization (ADRFLC) on the planar 2-DOF manipulator.

The ESO is built on the linearized "free" model; a feedback-linearizing law
(M, C from the ideal model) maps the virtual input to joint torques.

Runs PyBullet headless (DIRECT) and shows the ESO estimates and trajectory tracking.
"""
import matplotlib.pyplot as plt
import numpy as np
import pybullet
import pybullet_utils.bullet_client as bc
from utils.simulation import simulate

from manipulators.planar_2dof_pybullet import PlanarManipulator2DOFPyBullet
from trajectory_generators.sinusonidal import Sinusoidal
from controllers.adrc_flc_controller import ADRFLController

Tp = 0.001
end = 5.0

kp, kd, p = 50., 14., 150.   # error-pole gains and triple ESO pole

# traj_gen = ConstantTorque(np.array([0., 1.0])[:, np.newaxis])
traj_gen = Sinusoidal(np.array([0., 1.]), np.array([2., 2.]), np.array([0., 0.]))
# traj_gen = Poly3(np.array([0., 0.]), np.array([pi/4, pi/6]), end)

q0, qdot0, _ = traj_gen.generate(0.)

controller = ADRFLController(Tp, np.concatenate([q0, qdot0]),
                             np.diag([kp, kp]), np.diag([kd, kd]), np.array([p, p]))

manip = PlanarManipulator2DOFPyBullet(Tp, q0, qdot0)
T = np.linspace(0., end, int(end / Tp))
Q, Q_d, u, T = simulate("PYBULLET", traj_gen, controller, Tp, end)

eso = np.array(controller.eso.states)
plt.figure()
plt.subplot(321); plt.plot(T, Q[:, 0], 'r'); plt.plot(T, eso[:, 0], 'b--'); plt.title("q1")
plt.subplot(322); plt.plot(T, Q[:, 2], 'r'); plt.plot(T, eso[:, 2], 'b--'); plt.title("dq1")
plt.subplot(323); plt.plot(T, Q[:, 1], 'r'); plt.plot(T, eso[:, 1], 'b--'); plt.title("q2")
plt.subplot(324); plt.plot(T, Q[:, 3], 'r'); plt.plot(T, eso[:, 3], 'b--'); plt.title("dq2")
plt.subplot(325); plt.plot(T, Q[:, 0], 'r', label="q1"); plt.plot(T, Q_d[:, 0], 'b', label="q1_d"); plt.title("joint 1"); plt.xlabel("t [s]"); plt.legend()
plt.subplot(326); plt.plot(T, Q[:, 1], 'r', label="q2"); plt.plot(T, Q_d[:, 1], 'b', label="q2_d"); plt.title("joint 2"); plt.xlabel("t [s]"); plt.legend()
plt.tight_layout()
plt.show()
