import numpy as np

from .manipulator_model import ManiuplatorModel


class IdealModel(ManiuplatorModel):
    def __init__(self, Tp, m3=0.0, r3=0.01):
        super().__init__(Tp, m3, r3)
