from patternSearch import PatternSearch
from collections.abc import Callable
import numpy as np


class HookeJeevesMethod(PatternSearch):
    """
    Base class for all descent methods.
    Generally require
    - an initial point to start descending from
    - a method to determine the direction to descend
    - a method to determine how far to move in that direction
    """

    def __init__(self, initial_point, objective_function: Callable, objective_function_args: dict = {},
                 alpha=1, beta=0.5, max_function_calls=50, tolerance=0.001, opportunistic=True,
                 multithreaded=False):
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
        D = []
        for i in range(len(initial_point)):
            a = np.zeros(len(initial_point))
            a[i] = 1
            D.append(a.copy())
            D.append(-a)

        super().__init__(initial_point, objective_function, objective_function_args, D, alpha, beta, max_function_calls,
                         tolerance, opportunistic, multithreaded)
