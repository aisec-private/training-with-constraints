import numpy as np


class Box:
    def __init__(self, a, b):
        self.a = a
        self.b = b
        self.name = 'box'

    def is_empty(self):
        return np.any(self.a > self.b)

    def project(self, x):
        return np.clip(x, self.a, self.b)

    def sample(self):
        return (self.b - self.a) * np.random.random_sample(size=self.a.shape) + self.a
