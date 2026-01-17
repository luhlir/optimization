from firstOrderDescentMethod import FirstOrderDescentMethod
from architecture_temp.lineSearch.lineSearchEnum import LineSearchEnum
from collections.abc import Callable
from enum import Enum
import numpy as np


def polak_ribiere_beta_method(gradient, previous_gradient):
    return max(np.dot(gradient, gradient - previous_gradient) / np.dot(previous_gradient, previous_gradient), 0)


def fletcher_reeves_beta_method(gradient, previous_gradient):
    return np.dot(gradient, gradient) / np.dot(previous_gradient, previous_gradient)


class BetaMethodEnum(Enum):
    POLAK_RIBIERE = polak_ribiere_beta_method
    FLETCHER_REEVES = fletcher_reeves_beta_method


class ConjugateDescent(FirstOrderDescentMethod):

    def __init__(self, initial_point, objective_function: Callable, objective_function_args: dict = {},
                 gradient_function: Callable | None = None, gradient_function_args: dict = {},
                 use_auto_differentiation: bool = True,
                 line_search_method: LineSearchEnum = LineSearchEnum.STRONG_BACKTRACK_SEARCH, line_search_args: dict = {},
                 beta_method: BetaMethodEnum = BetaMethodEnum.POLAK_RIBIERE, tolerance=0.00001, max_function_calls=50):
        super().__init__(objective_function, objective_function_args, gradient_function,
                         gradient_function_args, use_auto_differentiation)
        self.initial_point = initial_point
        self.beta_method = beta_method
        self.tolerance = tolerance
        self.max_function_calls = max_function_calls
        line_search_args["objective_function"] = self._call_objective_function
        line_search_args["gradient_function"] = self._call_gradient_function
        self.line_search_method = line_search_method.value(**line_search_args)

    def optimize(self):
        x_curr = np.array(self.initial_point)
        x_prev = np.ones(len(self.initial_point)) * np.inf
        d_prev = np.zeros(len(self.initial_point))
        previous_gradient = np.ones(len(self.initial_point))

        while self.max_function_calls > self._function_calls and np.linalg.norm(x_curr - x_prev) > self.tolerance:
            x_prev = x_curr.copy()

            gradient = self._call_gradient_function(x_curr)

            beta = self.beta_method.__call__(gradient, previous_gradient)
            d = -gradient + beta * d_prev

            d_prev = d.copy()
            previous_gradient = gradient.copy()
            x_curr = self.line_search_method.search(x_curr, d)
        return x_curr

    # This should implement getting specific parameters as well as doing the steps