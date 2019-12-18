import numpy as np
import matplotlib.pyplot as plt


class Grid:
    def __init__(self, d, l, spawn_pos, end_pos, eps=0, closed=True):
        self.d = d
        self.l = l
        self.spawn_pos = spawn_pos
        self.end_pos = end_pos
        self.closed = closed
        self.eps = eps

        self.pos = spawn_pos

    def step(self, action):

        # Env randomness
        if np.random.rand() < self.eps:
            dir = np.random.randint(0, self.d)
            sign = np.random.randint(0,2)
        else:
            assert -self.d <= action <= self.d and action != 0
            dir = abs(action)-1
            if action > 0:
                sign = 1
            else:
                sign = -1

        # Changing position
        if self.closed:
            self.pos[dir] = min(self.l-1, max(self.pos[dir] + sign, 0))
        else:
            self.pos[dir] = (self.pos + sign )% self.l

        # Return status
        if self.pos == self.end_pos:
            return self.pos.copy(), True
        else:
            return self.pos.copy(), False

    def reset(self):
        self.pos = self.spawn_pos
        return self.pos.copy()


def render(positions, l):
    assert positions.shape[1] == 2
    plt.scatter(positions[:, 0], positions[:, 1], c=range(positions.shape[0]))
    plt.xlim(-1, l)
    plt.ylim(-1, l)
    plt.show()
