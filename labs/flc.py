import matplotlib.pyplot as plt
import numpy as np

from controllers.dummy_controller import DummyController
from controllers.feedback_linearization_controller import FeedbackLinearizationController
from trajectory_generators.constant_torque import ConstantTorque
from trajectory_generators.sinusonidal import Sinusoidal
from trajectory_generators.poly3 import Poly3
from utils.simulation import simulate

Tp = 0.01
start = 0
end = 3

"""
Switch between the feedback linearization controller and the dummy controller here.
"""
controller = FeedbackLinearizationController(Tp)
# controller = DummyController(Tp)

"""
Here you have some trajectory generators. You can use them to check your implementations.
"""
# traj_gen = ConstantTorque(np.array([0., 1.0])[:, np.newaxis])
# traj_gen = Sinusoidal(np.array([0., 1.]), np.array([2., 2.]), np.array([0., 0.]))
traj_gen = Poly3(np.array([0., 0.]), np.array([np.pi / 4, np.pi / 6]), end)


Q, Q_d, u, T = simulate("PYBULLET", traj_gen, controller, Tp, end)




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
