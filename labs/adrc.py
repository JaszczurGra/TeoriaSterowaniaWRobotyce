import matplotlib.pyplot as plt
import numpy as np
from numpy import pi

from controllers.adrc_controller import ADRController
from controllers.pd_controller import PDDecentralizedController
from trajectory_generators.constant_torque import ConstantTorque
from trajectory_generators.sinusonidal import Sinusoidal
from trajectory_generators.poly3 import Poly3
from utils.simulation import simulate

Tp = 0.001
end = 5.

b_est_1 = 6.85
b_est_2 = 54.8
kp_est_1 = 50.
kp_est_2 = 50.
kd_est_1 = 14.
kd_est_2 = 14.
p1 = 150.
p2 = 150.


# traj_gen = ConstantTorque(np.array([0., 1.0])[:, np.newaxis])
traj_gen = Sinusoidal(np.array([0., 1.]), np.array([2., 2.]), np.array([0., 0.]))
# traj_gen = Poly3(np.array([0., 0.]), np.array([pi/4, pi/6]), end)

q0, qdot0, _ = traj_gen.generate(0.)
q1_0 = np.array([q0[0], qdot0[0]])
q2_0 = np.array([q0[1], qdot0[1]])
params = [[b_est_1, kp_est_1, kd_est_1, p1, q1_0],
          [b_est_2, kp_est_2, kd_est_2, p2, q2_0]]


controller = ADRController(Tp, params)
# controller = ADRController(Tp, params, model_b=True)
# controller = PDDecentralizedController(np.array([kp_est_1, kp_est_2]), np.array([kd_est_1, kd_est_2]))

Q, Q_d, u, T = simulate("PYBULLET", traj_gen, controller, Tp, end)

eso1 = np.array(controller.joint_controllers[0].eso.states)
eso2 = np.array(controller.joint_controllers[1].eso.states)

plt.subplot(221)
plt.plot(T, eso1[:, 0])
plt.plot(T, Q[:, 0], 'r')
plt.subplot(222)
plt.plot(T, eso1[:, 1])
plt.plot(T, Q[:, 2], 'r')
plt.subplot(223)
plt.plot(T, eso2[:, 0])
plt.plot(T, Q[:, 1], 'r')
plt.subplot(224)
plt.plot(T, eso2[:, 1])
plt.plot(T, Q[:, 3], 'r')
plt.show()

plt.subplot(221)
plt.plot(T, Q[:, 0], 'r')
plt.plot(T, Q_d[:, 0], 'b')
plt.legend()


plt.subplot(222)
plt.plot(T, Q[:, 1], 'r')
plt.plot(T, Q_d[:, 1], 'b')
plt.legend()

plt.subplot(223)
plt.plot(T, u[:, 0], 'r')
plt.plot(T, u[:, 1], 'b')
plt.legend()


plt.show()



plt.subplot(321)
plt.plot(T, Q[:, 0], 'r', label='q')
plt.plot(T, Q_d[:, 0], 'b', label='q desired')
plt.title('joint 1 position')
plt.ylabel('q [rad]')
plt.legend()
plt.subplot(322)
plt.plot(T, Q[:, 1], 'r', label='q')
plt.plot(T, Q_d[:, 1], 'b', label='q desired')
plt.title('joint 2 position')
plt.legend()


plt.subplot(323)
plt.plot(T, Q[:, 2], 'r', label='q_dot')
plt.plot(T, Q_d[:, 2], 'b', label='q_dot desired')
plt.title('joint 1 velocity')
plt.ylabel('q_dot')
plt.legend()
plt.subplot(324)
plt.plot(T, Q[:, 3], 'r', label='q_dot')
plt.plot(T, Q_d[:, 3], 'b', label='q_dot desired')
plt.title('joint 2 velocity')
plt.legend()

plt.subplot(325)
plt.plot(T, u[:, 0], 'r', label='u')
plt.title('joint 1 control')
plt.xlabel('time ')
plt.ylabel('u ')
plt.legend()
plt.subplot(326)
plt.plot(T, u[:, 1], 'r', label='u')
plt.title('joint 2 control')
plt.xlabel('time ')
plt.legend()

plt.tight_layout()
plt.show()
