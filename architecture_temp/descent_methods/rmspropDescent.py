from firstOrderDescentMethod import FirstOrderDescentMethod
from collections.abc import Callable
import numpy as np


class RmspropDescent(FirstOrderDescentMethod):

    def __init__(self, initial_point, objective_function: Callable, objective_function_args: dict = {},
                 gradient_function: Callable | None = None, gradient_function_args: dict = {},
                 use_auto_differentiation: bool = True, alpha=0.01, gamma=0.9, tolerance=0.0001, max_function_calls=50):
        super().__init__(objective_function, objective_function_args, gradient_function,
                         gradient_function_args, use_auto_differentiation)
        self.initial_point = initial_point
        self.alpha = alpha
        self.gamma = gamma
        self.tolerance = tolerance
        self.max_function_calls = max_function_calls

    def optimize(self):
        s = np.zeros(len(self.initial_point))
        x_prev = np.ones(len(self.initial_point)) * np.inf
        x_curr = np.array(self.initial_point)

        while self.max_function_calls > self._function_calls and np.linalg.norm(x_curr - x_prev) > self.tolerance:
            x_prev = x_curr.copy()

            gradient = self._call_gradient_function(x_curr)

            s = self.gamma * s + (1 - self.gamma) * (gradient * gradient)
            x_curr = x_prev - (self.alpha / (10 ** -8 + np.sqrt(s))) * gradient

        return x_curr
