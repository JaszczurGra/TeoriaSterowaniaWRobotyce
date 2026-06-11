import numpy as np
from lab1_feedback_linearization.trajectory_generators.trajectory_generator import TrajectoryGenerator


class Poly3(TrajectoryGenerator):
    def __init__(self, start_q, desired_q, T):
        self.T = T
        self.q_0 = start_q
        self.q_k = desired_q
        # Boundary conditions: q(0)=q_0, q(T)=q_k, q_dot(0)=q_dot(T)=0 (eq. 29).
        # In the Bernstein-like basis below this gives the closed form:
        self.a_0 = self.q_0
        self.a_1 = 3 * self.q_0
        self.a_2 = 3 * self.q_k
        self.a_3 = self.q_k

    def generate(self, t):
        """
        3rd degree polynomial from q_0 to q_k with zero start/end velocity, plus its
        first and second derivatives (eq. 28-31). Derivatives w.r.t. normalized time are
        rescaled by 1/T and 1/T**2 (chain rule, since t was normalized to [0, 1]).
        """
        t /= self.T
        q = self.a_3 * t**3 + self.a_2 * t**2 * (1 - t) + self.a_1 * t * (1 - t)**2 + self.a_0 * (1 - t)**3
        q_dot = 3 * self.a_3 * t**2 + self.a_2 * (2 * t * (1 - t) - t**2) \
            + self.a_1 * ((1 - t)**2 - 2 * t * (1 - t)) - 3 * self.a_0 * (1 - t)**2
        q_ddot = 6 * self.a_3 * t + self.a_2 * (2 - 6 * t) \
            + self.a_1 * (6 * t - 4) + 6 * self.a_0 * (1 - t)
        return q, q_dot / self.T, q_ddot / self.T**2
