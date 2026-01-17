from abc import ABC
from firstOrderDescentMethod import FirstOrderDescentMethod
from architecture_temp.utils.hessianGroup import HessianGroup
from collections.abc import Callable


class SecondOrderDescentMethod(FirstOrderDescentMethod, ABC):
    """
    Base class for all descent methods.
    Generally require
    - an initial point to start descending from
    - a method to determine the direction to descend
    - a method to determine how far to move in that direction
    """

    def __init__(self, objective_function: Callable, objective_function_args: dict = {},
                 gradient_function: Callable | None = None, gradient_function_args: dict = {},
                 hessian_function: Callable | None = None, hessian_function_args: dict = {},
                 use_auto_differentiation: bool = True):
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
        super().__init__(objective_function, objective_function_args, gradient_function, gradient_function_args,
                         use_auto_differentiation)
        self.hessian_function = hessian_function
        self.hessian_function_args = hessian_function_args
        self.flatten_gradient = False

    def _call_hessian_function(self, x):
        """
        Determines the gradient of the objective function at x.
        If auto_differentiation is True, calls the objective function with a GradientGroup
        Else calls the provided gradient function
        This does count towards the total function calls performed.
        :param x: the point at which to determine the gradient
        """
        if self.use_auto_differentiation:
            return self._call_objective_function(HessianGroup.make_hessian_groups(x)).hessian
        else:
            self._function_calls += 1
            return self.hessian_function(x, **self.hessian_function_args)
