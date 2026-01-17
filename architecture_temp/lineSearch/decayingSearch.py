from lineSearch import LineSearch
from collections.abc import Callable


class DecayingSearch(LineSearch):

    def __init__(self, objective_function: Callable, alpha, decay_rate):
        super().__init__(objective_function)
        self.alpha = alpha
        self.decay_rate = decay_rate
        self.steps = 0

    def __search(self, x, d):
        result = self.alpha * (self.decay_rate ** self.steps)
        self.steps += 1
        return result
    