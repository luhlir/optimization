from collections.abc import Callable
from abc import ABC, abstractmethod
from multiprocessing import Lock


class Optimizer(ABC):
    """
    Base class for optimization method classes
    """

    def __init__(self, objective_function: Callable, objective_function_args: dict = {}):
        """
        Creates an instance of an Optimizer class
        :param objective_function: function to be minimized
        :param objective_function_args: dictionary of immutable arguments for the objective function
        """
        self.objective_function = objective_function
        self.objective_function_args = objective_function_args

        # We may be doing multithreaded batch evaluation of our objective function, in which case we should use a
        # mutex to make sure we are counting our function calls appropriately
        self.lock = Lock()
        self._function_calls = 0
        # TODO: _function_calls may be referenced in a multithreaded context and should be protected/atomic

    def _call_objective_function(self, x):
        """
        Wrapper for calling the objective function with a given input. Tracks the number of function calls to limit
        calls based on user provided values in the lower level optimization classes
        :param x: the point at which to assess the objective function
        :return: the value of the objective function at x
        """
        with self.lock:
            self._function_calls += 1
        return self.objective_function(x, **self.objective_function_args)

    @abstractmethod
    def optimize(self):
        pass

    def set_objective_function(self, function: Callable):
        """
        Sets a new objective function for the optimizer
        :param function: the new objective function to use
        """
        self.objective_function = function

    def get_objective_function(self) -> Callable:
        """
        Gets the objective function for the optimizer
        :return: the objective function, no arguments
        """
        return self.objective_function

    def set_objective_function_arguments(self, function_args: dict):
        """
        Sets new arguments for the objective function
        """
        self.objective_function_args = function_args

    def get_objective_function_arguments(self):
        """
        Gets the arguments for the objective function
        """
        return self.objective_function_args
