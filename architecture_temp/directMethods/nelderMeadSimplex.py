from architecture_temp.optimizer import Optimizer
from collections.abc import Callable
from architecture_temp.utils.spanningSet import make_minimal_positive_spanning_set
import numpy as np
from architecture_temp.utils.batchEvaluation import batch_eval


class NelderMeadSimplex(Optimizer):
    """
    Base class for all descent methods.
    Generally require
    - an initial point to start descending from
    - a method to determine the direction to descend
    - a method to determine how far to move in that direction
    """

    def __init__(self, initial_points, objective_function: Callable, objective_function_args: dict = {},
                 side_length=1, reflect=1, expansion=2, contraction=0.5, shrinkage=0.5,
                 max_function_calls=50, tolerance=0.001, multithreaded=False):
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

        # Create a set of simplex vertices unless one was passed in at the beginning
        self.points = np.array(initial_points)
        if len(self.points.shape) == 1:
            self.points = NelderMeadSimplex.make_regular_simplex(self.points, side_length)

        # Save parameters for the optimization
        self.reflect = reflect
        self.expansion = expansion
        self.contraction = contraction
        self.shrinkage = shrinkage
        self.tolerance = tolerance
        self.max_function_calls = max_function_calls
        self.multithreaded = multithreaded

    @staticmethod
    def make_regular_simplex(initial_point, side_length):
        n = len(initial_point)
        S = np.zeros((n + 1, n))
        one = np.ones(n)
        scale = np.array(initial_point) - (side_length / (n * np.sqrt(2))) * (1 + (1 / np.sqrt(n + 1))) * one
        for i in range(n):
            basis = np.zeros(n)
            basis[i] = side_length / np.sqrt(2)
            S[i] = basis + scale
        S[n] = (side_length / np.sqrt(2 * (n + 1))) * one + np.array(initial_point)
        return S

    def optimize(self):

        # Let the simplex crawl across the space
        Y = batch_eval(self._call_objective_function, self.points, self.multithreaded)
        while self.max_function_calls > self._function_calls and np.std(Y) > self.tolerance:
            indices = np.argsort(Y)
            self.points, Y = self.points[indices], Y[indices]
            x_avg = np.mean(self.points[:-1], 0)
            xr = x_avg + self.reflect * (x_avg - self.points[-1])
            yr = self._call_objective_function(xr)
            if yr < Y[0]:  # Go even further
                xe = x_avg + self.expansion * (xr - x_avg)
                ye = self._call_objective_function(xe)
                if ye < yr:  # Was expansion worth it?
                    self.points[-1], Y[-1] = xe.copy(), ye
                else:
                    self.points[-1], Y[-1] = xr.copy(), yr
            elif yr < Y[-2]:  # Contraction isn't going to help
                self.points[-1], Y[-1] = xr.copy(), yr
            else:  # Let's do a contraction
                if yr < Y[-1]:
                    self.points[-1], Y[-1] = xr.copy(), yr
                xc = x_avg + self.contraction * (self.points[-1] - x_avg)
                yc = self._call_objective_function(xc)
                if yc > Y[-1]:  # We will do a shrinkage instead
                    for i in range(len(self.points)):
                        self.points[i] = (self.points[i] + self.points[0]) * self.shrinkage
                    Y = batch_eval(self._call_objective_function, self.points, self.multithreaded)
                else:
                    self.points[-1], Y[-1] = xc.copy(), yc
        return self.points[np.argmin(Y)]
