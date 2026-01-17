from architecture_temp.optimizer import Optimizer
from collections.abc import Callable
from architecture_temp.utils.spanningSet import make_minimal_positive_spanning_set
import numpy as np
from architecture_temp.utils.batchEvaluation import batch_eval


class PatternSearch(Optimizer):
    """
    Base class for all descent methods.
    Generally require
    - an initial point to start descending from
    - a method to determine the direction to descend
    - a method to determine how far to move in that direction
    """

    def __init__(self, initial_point, objective_function: Callable, objective_function_args: dict = {},
                 D=None, alpha=1, beta=0.5, max_function_calls=50, tolerance=0.001, opportunistic=True,
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
        super().__init__(objective_function, objective_function_args)
        self.initial_point = initial_point
        self.tolerance = tolerance
        self.max_function_calls = max_function_calls
        if D is None:
            self.D = make_minimal_positive_spanning_set(1, len(self.initial_point))
        else:
            self.D = D
        self.alpha = alpha
        self.beta = beta
        self.opportunistic = opportunistic
        self.multithreaded = multithreaded

    def optimize(self):
        x_min = self.initial_point.copy()

        f_min = self._call_objective_function(x_min)
        alpha = self.alpha
        while self.max_function_calls > self._function_calls and alpha > self.tolerance:
            if self.opportunistic:
                for i in range(len(self.D)):
                    x = x_min + alpha * self.D[i]
                    y = self._call_objective_function(x)
                    if y < f_min:
                        x_min = x.copy()
                        f_min = y
                        # Move this direction to the top of the directions
                        d = self.D[i].copy()
                        for j in range(i):
                            d_temp = self.D[j].copy()
                            self.D[j] = d.copy()
                            d = d_temp.copy()
                        self.D[i] = d.copy()
                        print(x_min)
                        break
                else:
                    alpha *= self.beta
            else:
                X = self.D.copy()
                for i in range(len(self.D)):
                    X[i] = x_min + alpha * self.D[i]
                Y = batch_eval(self._call_objective_function, X, self.multithreaded)
                if np.min(Y) < f_min:
                    x_min = X[np.argmin(Y)].copy()
                    f_min = np.min(Y)
                else:
                    alpha *= self.beta
        return x_min
