from firstOrderDescentMethod import FirstOrderDescentMethod
from architecture_temp.lineSearch.lineSearchEnum import LineSearchEnum
from collections.abc import Callable
import numpy as np
from enum import Enum


def davidson_fletcher_powell_hessian_approximation(Q, gamma, delta):
    Q -= np.matmul(np.matmul(np.matmul(Q, gamma), np.transpose(gamma)), Q) / \
         np.dot(np.matmul(np.transpose(gamma), Q), gamma)
    Q += np.matmul(delta, np.transpose(delta)) / np.dot(delta.flatten(), gamma)
    return Q

def broyden_fletcher_goldfarb_shanno_hessian_approximation(Q, gamma, delta):
    delta_f = delta.flatten()
    temp1 = (np.matmul(np.matmul(delta, np.transpose(gamma)), Q) +
            np.matmul(np.matmul(Q, gamma), np.transpose(delta))) / np.dot(delta_f, gamma)
    temp2 = (1 + np.dot(np.matmul(np.transpose(gamma), Q), gamma) / np.dot(delta_f, gamma)) * \
            np.matmul(delta, np.transpose(delta)) / np.dot(delta_f, gamma)
    Q += temp2 - temp1
    return Q


class HessianApproximation(Enum):
    DAVIDSON_FLETCHER_POWELL = davidson_fletcher_powell_hessian_approximation
    BROYDEN_FLETCHER_GOLDFARB_SHANNO = broyden_fletcher_goldfarb_shanno_hessian_approximation


class QuasiNewtonsMethod(FirstOrderDescentMethod):

    def __init__(self, initial_point, objective_function: Callable, objective_function_args: dict = {},
                 gradient_function: Callable | None = None, gradient_function_args: dict = {},
                 use_auto_differentiation: bool = True,
                 hessian_approximation: HessianApproximation = HessianApproximation.DAVIDSON_FLETCHER_POWELL,
                 tolerance: float = 0.0001, max_function_calls: int = 500,
                 line_search_method: LineSearchEnum = LineSearchEnum.STRONG_BACKTRACK_SEARCH,
                 line_search_args: dict = {}):
        super().__init__(objective_function, objective_function_args, gradient_function,
                         gradient_function_args, use_auto_differentiation)
        self.hessian_approximation_method = hessian_approximation
        self.initial_point = initial_point
        self.tolerance = tolerance
        self.max_function_calls = max_function_calls
        line_search_args["objective_function"] = self._call_objective_function
        line_search_args["gradient_function"] = self._call_gradient_function
        self.line_search_method = line_search_method.value(**line_search_args)

    def optimize(self):
        x_curr = np.array(self.initial_point)
        x_prev = np.ones(len(self.initial_point)) * np.inf
        Q = np.identity(len(self.initial_point))

        grad = self._call_gradient_function(x_curr)

        while self.max_function_calls > self._function_calls and np.linalg.norm(x_curr - x_prev) > self.tolerance:
            x_prev = x_curr.copy()
            grad_prev = grad.copy()

            d = -np.matmul(Q, grad).flatten()
            x_curr = self.line_search_method.search(x_curr, d)

            grad = self._call_gradient_function(x_curr)

            gamma = grad - grad_prev
            delta = (x_curr - x_prev).reshape((len(self.initial_point), 1))
            Q = self.hessian_approximation_method.__call__(Q, gamma, delta)

        return x_curr

    # This should implement getting specific parameters as well as doing the steps
