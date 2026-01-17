from secondOrderDescentMethod import SecondOrderDescentMethod
from architecture_temp.lineSearch.lineSearchEnum import LineSearchEnum
from collections.abc import Callable
import numpy as np


class NewtonsMethod(SecondOrderDescentMethod):

    def __init__(self, initial_point, objective_function: Callable, objective_function_args: dict = {},
                 gradient_function: Callable | None = None, gradient_function_args: dict = {},
                 hessian_function: Callable | None = None, hessian_function_args: dict = {},
                 use_auto_differentiation: bool = True, tolerance: float = 0.0001, max_function_calls: int = 500,
                 line_search_method: LineSearchEnum = LineSearchEnum.STRONG_BACKTRACK_SEARCH,
                 line_search_args: dict = {}):
        super().__init__(objective_function, objective_function_args, gradient_function,
                         gradient_function_args, hessian_function, hessian_function_args, use_auto_differentiation)
        self.initial_point = initial_point
        self.tolerance = tolerance
        self.max_function_calls = max_function_calls
        line_search_args["objective_function"] = self._call_objective_function
        line_search_args["gradient_function"] = self._call_gradient_function
        self.line_search_method = line_search_method.value(**line_search_args)

    def optimize(self):
        x_curr = np.array(self.initial_point)
        x_prev = np.ones(len(self.initial_point)) * np.inf

        while self.max_function_calls > self._function_calls and np.linalg.norm(x_curr - x_prev) > self.tolerance:
            x_prev = x_curr.copy()

            grad = self._call_gradient_function(x_curr)
            hess = self._call_hessian_function(x_curr)

            d = -np.matmul(np.linalg.inv(hess + np.identity(len(hess))) * 0.00001, grad).flatten()
            x_curr = self.line_search_method.search(x_curr, d)

        return x_curr

    # This should implement getting specific parameters as well as doing the steps
