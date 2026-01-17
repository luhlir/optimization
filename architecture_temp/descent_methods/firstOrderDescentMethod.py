from abc import ABC
from architecture_temp.optimizer import Optimizer
from architecture_temp.utils.gradientGroup import GradientGroup
from collections.abc import Callable


class FirstOrderDescentMethod(Optimizer, ABC):
    """
    Base class for all descent methods.
    Generally require
    - an initial point to start descending from
    - a method to determine the direction to descend
    - a method to determine how far to move in that direction
    """

    def __init__(self, objective_function: Callable, objective_function_args: dict = {},
                 gradient_function: Callable | None = None, gradient_function_args: dict = {},
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
        super().__init__(objective_function, objective_function_args)
        self.gradient_function = gradient_function
        self.gradient_function_args = gradient_function_args
        self.use_auto_differentiation = use_auto_differentiation
        self.flatten_gradient = True

    def _call_gradient_function(self, x):
        """
        Determines the gradient of the objective function at x.
        If auto_differentiation is True, calls the objective function with a GradientGroup
        Else calls the provided gradient function
        This does count towards the total function calls performed.
        :param x: the point at which to determine the gradient
        """
        if self.use_auto_differentiation:
            gradient = self._call_objective_function(GradientGroup.make_gradient_groups(x)).gradients
            if self.flatten_gradient:
                return gradient.flatten()
            else:
                return gradient
        else:
            self._function_calls += 1
            return self.gradient_function(x, **self.gradient_function_args)

    def set_gradient_function(self, function: Callable):
        self.gradient_function = function

    def get_gradient_function(self):
        return self.gradient_function

    def set_gradient_arguments(self, arguments: dict):
        self.gradient_function_args = arguments

    def get_gradient_arguments(self) -> dict:
        return self.gradient_function_args

    def set_use_automatic_differentiation(self, use_auto_differentiation: bool):
        self.use_auto_differentiation = use_auto_differentiation

    def get_use_automatic_differentiation(self) -> bool:
        return self.use_auto_differentiation
