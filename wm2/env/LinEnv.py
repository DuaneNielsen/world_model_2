import numpy as np


class LinEnv:
    def __init__(self):
        self.pos = np.array([0.0], dtype=np.float32)
        self.step_count = 0

    def reset(self):
        self.pos = np.array([0.0], dtype=np.float32)
        self.step_count = 0
        return self.pos.copy()

    def step(self, action):
        self.step_count += 1
        self.pos += action
        pos = self.pos.copy()

        if pos >= 2.0:
            return pos, 1.0, True
        elif self.step_count > 10:
            return pos, 0.0, True
        elif pos <= -2.0:
            return pos, -1.0, True
        else:
            return pos, 0.0, False