from lineSearch import LineSearch
from gradientGroup import GradientGroup as gg
import numpy as np
from collections.abc import Callable


class StrongBacktrackSearch(LineSearch):

    def __init__(self, objective_function: Callable, gradient_function: Callable, alpha=1, p=0.5, beta=0.0001,
                 sigma=0.1, max_steps=20):
        super().__init__(objective_function)
        self.gradient_function = gradient_function
        self.alpha = alpha
        if p > 1:
            self.p = 1 / p
        else:
            self.p = p
        self.beta = beta
        self.sigma = sigma
        self.max_steps = max_steps

    def __search(self, x, d):
        """
        Performs a backtrack search that uses the strong Wolfe conditions to detect a local minimum on the line

        :param f: function to evaluate
        :param x: starting point for search
        :param f_args: dictionary of immutable arguments to f
        :param d: direction vector to search in
        :param f_prime: gradient method of f
        :param fp_args: dictionary of immutable arguments to f_prime
        :param auto_diff: use automatic differentiation instead of an explicit gradient method
        :param alpha: initial step size
        :param p: step size correction factor <1 used at each step in bracketing phase
        :param beta: scalar for expected change in function value
        :param sig: scalar used in zoom phase
        :param max_steps: maximum number of function calls to make
        :return: scalar describing the step size to take in the given direction
        """
        point_value = self.objective_function(x)
        point_gradient = self.gradient_function(x)

        point_expected = np.dot(point_gradient, d)

        last_alpha = 0
        alpha = self.alpha
        max_steps = self.max_steps
        while max_steps > 0:
            next_step = x + alpha * d
            next_value = self.objective_function(next_step)
            next_gradient = self.gradient_function(next_step)

            next_expected = np.dot(next_gradient, d)

            if next_value >= point_value or next_value > point_value + self.beta * alpha * point_expected or next_expected >= 0:
                break
            else:
                last_alpha = alpha
                alpha *= self.p
            max_steps -= 1

        # We've bracketed the wolfe condition zone to [last_alpha, alpha]
        # Check Wolfe conditions for the endpoint alpha
        if next_value <= point_value + self.beta * alpha * point_expected and abs(next_expected) <= -self.sigma * point_expected:
            return alpha

        next_step = x + last_alpha * d
        next_value = self.objective_function(next_step)
        next_gradient = self.gradient_function(next_step)

        next_expected = np.dot(next_gradient, d)

        # Check Wolfe conditions for the  endpoint last_alpha
        if next_value <= point_value + self.beta * last_alpha * point_expected and abs(
                next_expected) <= -self.sigma * point_expected:
            return last_alpha

        midpoint = (alpha + last_alpha) / 2
        while max_steps > 0:
            midpoint = (alpha + last_alpha) / 2
            mid_step = x + midpoint * d

            mid_value = self.objective_function(mid_step)
            mid_gradient = self.gradient_function(mid_step)

            mid_expected = np.dot(mid_gradient, d)
            # Check Wolfe conditions for midpoint and return if met
            if mid_value <= point_value + self.beta * midpoint * point_expected and abs(
                    mid_expected) <= -self.sigma * point_expected:
                return midpoint

            # Check boundary conditions for midpoint
            if mid_value >= point_value or mid_value > point_value + self.beta * midpoint * point_expected or mid_expected >= 0:
                alpha = midpoint
            else:
                last_alpha = midpoint
            max_steps -= 1
        return midpoint
