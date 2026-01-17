from firstOrderDescentMethod import FirstOrderDescentMethod
from collections.abc import Callable
import numpy as np


class AdamDescent(FirstOrderDescentMethod):

    def __init__(self, initial_point, objective_function: Callable, objective_function_args: dict = {},
                 gradient_function: Callable | None = None, gradient_function_args: dict = {},
                 use_auto_differentiation: bool = True, alpha=0.001, delta=0.9, gamma=0.999, tolerance=0.0001, max_function_calls=50):
        super().__init__(objective_function, objective_function_args, gradient_function,
                         gradient_function_args, use_auto_differentiation)
        self.initial_point = initial_point
        self.alpha = alpha
        self.delta = delta
        self.gamma = gamma
        self.tolerance = tolerance
        self.max_function_calls = max_function_calls

    def optimize(self):
        s = np.zeros(len(self.initial_point))
        v = np.zeros(len(self.initial_point))
        x_prev = np.ones(len(self.initial_point)) * np.inf
        x_curr = np.array(self.initial_point)
        e = 10 ** -8
        k = 1

        while self.max_function_calls > self._function_calls and np.linalg.norm(x_curr - x_prev) > self.tolerance:
            x_prev = x_curr.copy()

            gradient = self._call_gradient_function(x_curr)

            v = self.delta * v + (1 - self.delta) * gradient
            s = self.gamma * s + (1 - self.gamma) * gradient * gradient
            dv = v / (1 - (self.delta ** k))
            ds = s / (1 - (self.gamma ** k))
            x_curr = x_prev - self.alpha * dv / (e + np.sqrt(ds))

            k += 1

        return x_curr
