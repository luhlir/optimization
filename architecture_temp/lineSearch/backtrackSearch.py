from lineSearch import LineSearch
from gradientGroup import GradientGroup as gg
import numpy as np
from collections.abc import Callable


class BacktrackSearch(LineSearch):

    def __init__(self, objective_function: Callable, gradient_function: Callable, alpha=1, p=0.5, beta=0.0001,
                 max_steps=20):
        super().__init__(objective_function)
        self.gradient_function = gradient_function
        self.alpha = alpha
        if p > 1:
            self.p = 1 / p
        else:
            self.p = p
        self.beta = beta
        self.max_steps = max_steps

    def __search(self, x, d):
        value = self.objective_function(x)
        gradient = self.gradient_function(x)

        point_expected = np.dot(gradient, d)

        max_steps = self.max_steps
        alpha_updated = self.alpha
        # TODO: Should both of the references to alpha below be alpha_updated??
        while self.objective_function(x + self.alpha * d) > value + self.beta * alpha_updated * point_expected \
                and max_steps > 0:
            alpha_updated *= self.p
            max_steps -= 1

        return alpha_updated
