from lineSearch import LineSearch
from collections.abc import Callable


class FixedSearch(LineSearch):

    def __init__(self, objective_function: Callable, alpha):
        super().__init__(objective_function)
        self.alpha = alpha

    def __search(self, x, d):
        return self.alpha
