from architecture_temp.optimizer import Optimizer
from collections.abc import Callable
import numpy as np
from architecture_temp.utils.batchEvaluation import BatchEval


class DividedRectangles(Optimizer):
    """
    Base class for all descent methods.
    Generally require
    - an initial point to start descending from
    - a method to determine the direction to descend
    - a method to determine how far to move in that direction
    """

    def __init__(self, interval_low, interval_high, objective_function: Callable, objective_function_args: dict = {},
                 max_function_calls=50, size_tolerance=0.001, value_tolerance=0.001, multithreaded=False):
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

        self.interval_low = interval_low
        self.interval_high = interval_high
        self.size_tolerance = size_tolerance
        self.value_tolerance = value_tolerance
        self.max_function_calls = max_function_calls
        self.batch_eval = BatchEval(multithreaded)

    def _unit_objective_function(self, x):
        return self._call_objective_function(x * np.subtract(self.interval_high, self.interval_low) + self.interval_low)

    def _interval_divide_wrapper(self, interval):
        return Interval.divide(interval, self._unit_objective_function)

    def optimize(self):

        n = len(self.interval_low)
        c = np.ones(n) * 0.5
        interval0 = Interval(np.zeros(n), c, self._unit_objective_function(c))
        widths = {interval0.key: [interval0]}
        best = interval0

        # Start the whole thing off!
        while self.max_function_calls > self._function_calls and best.vertex_dist() > self.size_tolerance:
            # Get the optimal intervals
            optimal = []
            empty = 0
            for key in widths:
                if len(widths[key]) > 0:
                    optimal.append(np.sort(widths[key])[0])
                else:
                    empty += 1
            optimal = sorted(optimal, key=lambda interval: interval.vertex_dist())
            if len(optimal) > 2:
                x, y = optimal[0].vertex_dist(), optimal[0].y
                x1, y1 = optimal[1].vertex_dist(), optimal[1].y
                i = 2
                while i < len(optimal):
                    # If the previous optimal sample is above the line between its neighbors, remove it
                    x2, y2 = optimal[i].vertex_dist(), optimal[i].y
                    slope = (y2 - y) / (x2 - x)
                    if y1 > slope * (x1 - x) + y + self.value_tolerance:
                        optimal.remove(optimal[i - 1])
                        x1, y1 = x2, y2
                    else:
                        i += 1
                        x, y = x1, y1
                        x1, y1 = x2, y2

            # Divide all of the potentially optimal points!
            args = []
            for interval in optimal:
                widths[interval.key].remove(interval)
                args.append((interval, self._unit_objective_function))
            new_interval_sets = self.batch_eval.eval(self._interval_divide_wrapper, args)
            for new_intervals in new_interval_sets:
                for interval in new_intervals:
                    if interval.key not in widths.keys():
                        widths[interval.key] = [interval]
                    else:
                        widths[interval.key].append(interval)
                best = min(min(new_intervals), best)

        return best.center * np.subtract(self.interval_high, self.interval_low) + self.interval_low


class Interval:
    def __init__(self, depths, center, y):
        self.depths = depths.copy()
        self.center = center.copy()
        self.y = y
        self.key = str(self.vertex_dist())

    def __lt__(self, other):
        return self.y < other.y

    def __gt__(self, other):
        return self.y > other.y

    def __le__(self, other):
        return self.y <= other.y

    def __ge__(self, other):
        return self.y >= other.y

    def __eq__(self, other):
        return self.y == other.y

    def vertex_dist(self):
        return np.linalg.norm(0.5 * (3 ** (-self.depths)))

    @staticmethod
    def divide(interval, f):
        depth = np.min(interval.depths)
        movement = 3 ** (-depth - 1)
        direction = np.argmin(interval.depths)
        new_depths = interval.depths.copy()
        new_depths[direction] += 1
        c0 = interval.center.copy()
        c2 = interval.center.copy()
        c0[direction] += movement
        c2[direction] -= movement
        int0 = Interval(new_depths, c0, f(c0))
        int1 = Interval(new_depths, interval.center, interval.y)
        int2 = Interval(new_depths, c2, f(c2))
        return [int0, int1, int2]
