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
from trajectory_generators.poly3 import Poly3
from trajectory_generators.constant_torque import ConstantTorque
from controllers.adrc_flc_controller import ADRFLController
from math import pi 

Tp = 0.001
end = 5.0

kp, kd, p = 50., 14., 150.   # error-pole gains and triple ESO pole

# traj_gen = ConstantTorque(np.array([0., 1.0])[:, np.newaxis])
# traj_gen = Sinusoidal(np.array([0., 1.]), np.array([2., 2.]), np.array([0., 0.]))
traj_gen = Poly3(np.array([0., 0.]), np.array([pi/4, pi/6]), end)

q0, qdot0, _ = traj_gen.generate(0.)

controller = ADRFLController(Tp, np.concatenate([q0, qdot0]),
                             np.diag([kp, kp]), np.diag([kd, kd]), np.array([p, p]))

Q, Q_d, u, T = simulate("PYBULLET", traj_gen, controller, Tp, end)

eso = np.array(controller.eso.states)
plt.figure()

plt.subplot(421)
plt.plot(T, Q[:, 0], 'r', label='q measured')
plt.plot(T, eso[:, 0], 'g', label='q estimated (ESO)')
plt.title('joint 1 position')
plt.ylabel('q [rad]')
plt.legend()

plt.subplot(422)
plt.plot(T, Q[:, 1], 'r', label='q measured')
plt.plot(T, eso[:, 1], 'g', label='q estimated (ESO)')
plt.title('joint 2 position')
plt.legend()

plt.subplot(423)
plt.plot(T, Q[:, 2], 'r', label='q_dot measured')
plt.plot(T, eso[:, 2], 'g', label='q_dot estimated (ESO)')
plt.title('joint 1 velocity')
plt.ylabel('q_dot [rad/s]')
plt.legend()

plt.subplot(424)
plt.plot(T, Q[:, 3], 'r', label='q_dot measured')
plt.plot(T, eso[:, 3], 'g', label='q_dot estimated (ESO)')
plt.title('joint 2 velocity')
plt.legend()

# Tracking: position vs desired
plt.subplot(425)
plt.plot(T, Q[:, 0], 'r', label='q')
plt.plot(T, Q_d[:, 0], 'b', label='q desired')
plt.title('joint 1 tracking')
plt.ylabel('q [rad]')
plt.legend()

plt.subplot(426)
plt.plot(T, Q[:, 1], 'r', label='q')
plt.plot(T, Q_d[:, 1], 'b', label='q desired')
plt.title('joint 2 tracking')
plt.legend()

plt.subplot(427)
plt.plot(T, u[:, 0], 'r', label='u')
plt.title('joint 1 control')
plt.xlabel('time [s]')
plt.ylabel('u [Nm]')
plt.legend()

plt.subplot(428)
plt.plot(T, u[:, 1], 'b', label='u')
plt.title('joint 2 control')
plt.xlabel('time [s]')
plt.legend()

plt.tight_layout()
plt.show()
