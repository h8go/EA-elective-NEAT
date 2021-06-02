import numpy as np
from minatar import Environment


class MinatarBreakoutWrapper(Environment):
    """

    """
    def state(self):
        state = super().state()
        state = state.transpose((2, 0, 1))
        state = np.sum([state[i] * (i+1) for i in range(state.shape[0])], axis=0)
        return state


class MinatarFreewaytWrapper(Environment):
    """

    """
    def state(self):
        state = super().state()
        state = state.transpose((2, 0, 1))
        state = np.sum([state[i] * (i+1) for i in range(state.shape[0])], axis=0)
        return state
