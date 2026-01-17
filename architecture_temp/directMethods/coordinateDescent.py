from architecture_temp.optimizer import Optimizer
from collections.abc import Callable
from architecture_temp.lineSearch.lineSearchEnum import LineSearchEnum
import numpy as np


class CoordinateDescent(Optimizer):
    """
    Base class for all descent methods.
    Generally require
    - an initial point to start descending from
    - a method to determine the direction to descend
    - a method to determine how far to move in that direction
    """

    def __init__(self, initial_point, objective_function: Callable, objective_function_args: dict = {},
                 accel=False, tolerance: float = 0.0001, max_function_calls: int = 500,
                 line_search_method: LineSearchEnum = LineSearchEnum.FULL_SEARCH, line_search_args: dict = {}):
        """
        :param initial_point: the initial point to start from when descending
        :param objective_function: function to be minimized
        :param objective_function_args: dictionary of immutable arguments for the objective function
        :param gradient_function: the gradient function of the objective function (can be None if auto_differentiation
            is True)
        :param gradient_function_args: dictionary of immutable arguments for the gradient function
        :param use_auto_differentiation: if True, gradient function will be ignored and automatic differentiation will
            be performed on the objective function to determine the gradient
        """
        super().__init__(objective_function, objective_function_args)
        self.initial_point = initial_point
        self.accel = accel
        self.tolerance = tolerance
        self.max_function_calls = max_function_calls
        line_search_args["objective_function"] = self._call_objective_function
        self.line_search = line_search_method.value(**line_search_args)

    def optimize(self):
        x_curr = np.array(self.initial_point)
        x_prev = np.ones(len(self.initial_point)) * np.inf
        while self.max_function_calls > self._function_calls and np.linalg.norm(x_curr - x_prev) > self.tolerance:
            x_prev = x_curr.copy()
            for i in range(len(self.initial_point)):
                d = np.zeros(len(self.initial_point))
                d[i] = 1
                x_curr = self.line_search.search(x_curr, d)
            if self.accel:
                diag = x_curr - np.array(self.initial_point)
                x_curr = self.line_search.search(x_curr, diag / np.linalg.norm(diag))

        return x_curr
