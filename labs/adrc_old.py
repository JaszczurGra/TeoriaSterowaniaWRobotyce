import matplotlib.pyplot as plt
import numpy as np
from numpy import pi
from scipy.integrate import odeint

from controllers.adrc_controller import ADRController

from trajectory_generators.constant_torque import ConstantTorque
from trajectory_generators.sinusonidal import Sinusoidal
from trajectory_generators.poly3 import Poly3
from utils.simulation import simulate

Tp = 0.001
end = 5

# traj_gen = ConstantTorque(np.array([0., 1.0])[:, np.newaxis])
traj_gen = Sinusoidal(np.array([0., 1.]), np.array([2., 2.]), np.array([0., 0.]))
# traj_gen = Poly3(np.array([0., 0.]), np.array([pi/4, pi/6]), end)

b_est_1 = 6.85
b_est_2 =  54.8
kp_est_1 = 50.
kp_est_2 = 50.
kd_est_1 = 14.
kd_est_2 = 14.
p1 = 150.
p2 = 150.

q0, qdot0, _ = traj_gen.generate(0.)
q1_0 = np.array([q0[0], qdot0[0]])
q2_0 = np.array([q0[1], qdot0[1]])
controller = ADRController(Tp, params=[[b_est_1, kp_est_1, kd_est_1, p1, q1_0],
                                       [b_est_2, kp_est_2, kd_est_2, p2, q2_0]])

Q, Q_d, u, T = simulate("PYBULLET", traj_gen, controller, Tp, end)

plt.figure()
for name, _ in controllers:
    T, Q, Q_d, _ = results[name]
    plt.subplot(211); plt.plot(T, Q[:, 0] - Q_d[:, 0], label=name)
    plt.subplot(212); plt.plot(T, Q[:, 1] - Q_d[:, 1], label=name)
plt.subplot(211); plt.title("joint 1 error [rad]"); plt.legend()
plt.subplot(212); plt.title("joint 2 error [rad]"); plt.xlabel("t [s]"); plt.legend()
plt.tight_layout()

T, Q, _, ctrl = results["ADRC"]
eso1 = np.array(ctrl.joint_controllers[0].eso.states)   # [q, q_dot, f]
eso2 = np.array(ctrl.joint_controllers[1].eso.states)
plt.figure()
plt.subplot(221); plt.plot(T, Q[:, 0], 'r'); plt.plot(T, eso1[:, 0], 'b--'); plt.title("q1")
plt.subplot(222); plt.plot(T, Q[:, 2], 'r'); plt.plot(T, eso1[:, 1], 'b--'); plt.title("dq1")
plt.subplot(223); plt.plot(T, Q[:, 1], 'r'); plt.plot(T, eso2[:, 0], 'b--'); plt.title("q2")
plt.subplot(224); plt.plot(T, Q[:, 3], 'r'); plt.plot(T, eso2[:, 1], 'b--'); plt.title("dq2")
plt.tight_layout()
plt.show()
