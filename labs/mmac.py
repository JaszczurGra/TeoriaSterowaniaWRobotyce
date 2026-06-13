import matplotlib.pyplot as plt
import numpy as np
from numpy import pi
from scipy.integrate import odeint

from controllers.dummy_controller import DummyController
from controllers.feedback_linearization_controller import FeedbackLinearizationController
from controllers.mma_controller import MMAController
from trajectory_generators.constant_torque import ConstantTorque
from trajectory_generators.sinusonidal import Sinusoidal
from trajectory_generators.poly3 import Poly3
from utils.simulation import simulate

"""http://www.gipsa-lab.fr/~ioandore.landau/adaptivecontrol/Transparents/Courses/AdaptiveCourse5GRK.pdf"""

Tp = 0.01
end = 3.


controller = MMAController(Tp)
# controller = FeedbackLinearizationController(Tp)
#controller = DummyController(Tp)

"""
Here you have some trajectory generators. You can use them to check your implementations.
"""
# traj_gen = ConstantTorque(np.array([0., 1.0])[:, np.newaxis])
traj_gen = Sinusoidal(np.array([0., 1.]), np.array([2., 2.]), np.array([0., 0.]))
# traj_gen = Poly3(np.array([0., 0.]), np.array([pi/4, pi/6]), end)


Q, Q_d, u, T, model_d, model = simulate("PYBULLET", traj_gen, controller, Tp, end, multimodel=True)

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

plt.subplot(3, 2, (5, 6))
plt.plot(T, model_d, 'r', label='model desired')
plt.plot(T, model, 'b', label='model true')
plt.title('model selection')
plt.xlabel('time ')
plt.ylabel('model index')
plt.legend()


plt.tight_layout()
plt.show()
