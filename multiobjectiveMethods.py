import numpy as np
from populationMethods import random_cauchy, random_normal, random_uniform
from zeroOrderMethods import batch_eval
import optimization


def dominates(y1, y2):
    """
    Determines if y1 is pareto dominant to y2

    :param y1: first array
    :param y2: second array
    :return: True if y1 dominates y2, else False
    """
    return all(y1[i] <= y2[i] for i in range(len(y1))) and any(y1[i] < y2[i] for i in range(len(y1)))


def scan_weights(weight_step, obj_count, scan_function, scan_args):
    """
    Performs a scan over a set of weights and finds function values at each step. Each weight array will add up to 1. function must take argument "weights".

    :param weight_step: distance between adjacent weight scan points
    :param obj_count: number of values needed in the weight array
    :param function: function that takes "weights" parameter
    :param args: mutable dictionary of arguments for function
    :return: unordered array of values returned by function at each step
    """
    weight_values = np.linspace(0, 1, num=int(1 / weight_step) + 1)
    d = obj_count - 1
    c = np.zeros(obj_count, dtype=int)
    ind = np.zeros(obj_count, dtype=int)
    ind[0] = len(weight_values) - 1
    points = [scan_function(**scan_args, weights=weight_values[ind])]
    while True:
        if d == obj_count - 1 or ind[0] == 0 or ind[0] < c[d]:
            ind[0] += ind[d]
            ind[d] = 0
            c[d] = 0
            d -= 1
            if d < 0:
                break
            c[d] += 1
        else:
            while d < obj_count - 1:
                ind[0] -= c[d]
                ind[d + 1] = c[d]
                d += 1
            points.append(scan_function(**scan_args, weights=weight_values[ind]))
    return np.array(points)


def naive_pareto(f, f_args, x_0=None, init_method="cauchy", variance=None,
                 lower=None, upper=None, sample_count=100, multithreaded=False):
    """
    Generates a Pareto frontier by evaluating random points and returning the ones that are Pareto dominant

    :param f: objective function
    :param f_args: dictionary of immutable arguments for the objective function
    :param x_0: starting point for normal or cauchy distribution population initialization
    :param init_method: method of population initialization around a point
    :param variance: covariance matrix of array of variances in each dimension based on initialization method
    :param lower: lower bounds used in uniform population initialization
    :param upper: upper bounds used in uniform population initialization
    :param sample_count: number of samples to use
    :param multithreaded: whether the population is evaluated in parallel
    :return: a list of points forming a Pareto frontier
    """
    if x_0 is not None:
        if init_method == "normal":
            samples = random_normal(sample_count, x_0, variance)
        else:
            samples = random_cauchy(sample_count, x_0, variance)
    else:
        samples = random_uniform(sample_count, lower, upper)
    objectives = batch_eval(f, samples, f_args, multithreaded)
    samples = list(samples)

    # Iterate through the samples and remove any that are nondominant
    i = 0
    while i < len(samples):
        if any(dominates(objectives[j], objectives[i]) for j in range(len(objectives))):
            objectives.pop(i)
            samples.pop(i)
        else:
            i += 1
    return np.array(samples)


# TODO: Make some constraint method magic. Might need refactoring a bit to incorporate already developed constraint code
# Straight-forward contraint method
# Lexicographic constraint method


def weighted_function(x, f, f_args, weights):
    """
    Weighted sum of the objectives

    :param x: design point
    :param f: objective function
    :param f_args: immutable dictionary of function arguments
    :param weights: array of weights
    :return: weighted sum of the objectives
    """
    return np.dot(f(x, **f_args), weights)


def weighted_pareto_method(f, x_0, f_args, weights,
                           nested_opt_method="newtons_method", nested_opt_args={}):
    """
    Optimizes a weighted sum of the objectives

    :param f: objective function
    :param x_0: initial design point
    :param f_args: immutable dictionary of objective function arguments
    :param weights: weights for objectives
    :param nested_opt_method: optimization method to use
    :param nested_opt_args: dictionary of arguments for optimization method
    :return: a likely locally minimal design point for given weights
    """
    wrapper_args = {'f': f,
                    'f_args': f_args,
                    'weights': weights}
    return optimization.optimize(weighted_function, x_0, wrapper_args, nested_opt_method, **nested_opt_args)


def weighted_pareto_scan_method(f, x_0, f_args, obj_count, weight_step=0.1,
                                nested_opt_method="newtons_method", nested_opt_args={}):
    """
    Varies a set of weights for each objective and optimizes the function at each weight

    :param f: objective function
    :param x_0: initial design point for each optimization
    :param f_args: dictionary of immutable arguments for the objective function
    :param obj_count: number of objectives expected in the objective function output
    :param weight_step: difference between sets of weights, smaller step -> more weights
    :param nested_opt_method: optimization method to run with each set of weights
    :param nested_opt_args: dictionary of immutable arguments for the optimization method
    :return: a numpy array of design points shaping a likely pareto frontier
    """
    scan_args = {"f": f,
                 "x_0": x_0,
                 "f_args": f_args,
                 "nested_opt_method": nested_opt_method,
                 "nested_opt_args": nested_opt_args}
    return scan_weights(weight_step, obj_count, weighted_pareto_method, scan_args)


def goal_wrapper(x, f, f_args, goal_obj, p):
    """
    Calculates the p-norm distance between a value and a goal value

    :param x: design point
    :param f: objective function
    :param f_args: immutable dictionary of objective function arguments
    :param goal_obj: goal objectives
    :param p: order of norm to use
    :return: p-norm distance from a goal objective
    """
    return np.linalg.norm(f(x, **f_args) - goal_obj, ord=p)


def goal_method(f, x_0, f_args, goal_obj, p=1, nested_opt_method="newtons_method", nested_opt_args={}):
    """
    Determines design point that results in the closest objectives to a goal objective

    :param f: objective function
    :param x_0: initial design point
    :param f_args: immutable dictionary of objective function arguments
    :param goal_obj: goal objective to optimize towards
    :param p: order of norm to use in distance calculation
    :param nested_opt_method: optimization method to use
    :param nested_opt_args: arguments for the optimization method
    :return: a design point likely resulting in objectives closest to goal
    """
    wrapper_args = {"f": f,
                    "f_args": f_args,
                    "goal_obj": goal_obj,
                    "p": p}
    return optimization.optimize(goal_wrapper, x_0, wrapper_args, nested_opt_method, **nested_opt_args)


def weighted_goal_wrapper(x, f, f_args, goal_obj, p):
    """
    Estimates the p-norm distance between a point and a goal objective

    :param x: design point
    :param f: objective function
    :param f_args: immutable dictionary of objective function arguments
    :param goal_obj: goal objective
    :param p: order of norm to estimate
    :return: estimated p-norm distance between a design point result and goal objective
    """
    return (f(x, **f_args) - goal_obj) ** p


def weighted_goal_method(f, x_0, f_args, goal_obj, weights, p=1,
                         nested_opt_method="newtons_method", nested_opt_args={}):
    """
    Optimizes a weighted p-norm distance from a goal objective

    :param f: objective function
    :param x_0: initial design point
    :param f_args: immutable dictionary of arguments for the objective function
    :param goal_obj: goal objective
    :param weights: weights to use for the p-norm distance
    :param p: order of norm to use for distance calculations
    :param nested_opt_method: optimization method to use
    :param nested_opt_args: arguments for the optimization method
    :return: a design point likely resulting in objectives closest to weighted goal
    """
    wrapper_args = {"f": f,
                    "f_args": f_args,
                    "goal_obj": goal_obj,
                    "p": p}
    return weighted_pareto_method(weighted_goal_wrapper, x_0, wrapper_args, weights,
                                  nested_opt_method, nested_opt_args)


def weighted_goal_scan_method(f, x_0, f_args, goal_obj, p=1, weight_step=0.1,
                              nested_opt_method="newtons_method", nested_opt_args={}):
    """
    Varies a set of weights for each objective and optimizes the p-norm distance from a goal objective

    :param f: objective function
    :param x_0: initial design point for each step
    :param f_args: immutable dictionary of argument for the objective function
    :param goal_obj: goal objective
    :param p: order of norm to use for distance calculations
    :param weight_step: step size between weight arrays
    :param nested_opt_method: optimization method to use at each step
    :param nested_opt_args: arguments for the optimization method
    :return: array of design points likely forming a pareto frontier
    """
    wrapper_args = {"f": f,
                    "f_args": f_args,
                    "goal_obj": goal_obj,
                    "p": p}
    return weighted_pareto_scan_method(weighted_goal_wrapper, x_0, wrapper_args, len(goal_obj), weight_step,
                                       nested_opt_method, nested_opt_args)


def min_max_wrapper(x, f, f_args, goal_obj, weights, scalar):
    """
    Calculates the maximum distance from a goal objective in any given dimension (infinite-norm distance). Adds an additional correction for use in optimization.

    :param x: design point
    :param f: objective function
    :param f_args: immutable dictionary of arguments for the objective function
    :param goal_obj: goal objective
    :param weights: weights for each objective
    :param scalar: small correction value (~0.0001 to 0.01)
    :return: maximum distance from a goal objective in any given dimension
    """
    real_goal = np.ones(len(goal_obj))
    value = f(x, **f_args) + real_goal - goal_obj   # Shift the objective function so real goal is all positive
    return max(weights * (value - real_goal)) + scalar * np.dot(value, real_goal)


def weighted_min_max_method(f, x_0, f_args, goal_obj, weights, scalar=0.01,
                            nested_opt_method="newtons_method", nested_opt_args={}):
    """
    Optimizes the weighted infinite-norm distance from a goal objective

    :param f: objective function
    :param x_0: initial design point
    :param f_args: immutable dictionary of arguments for the objective function
    :param goal_obj: goal objective
    :param weights: weights for each objective
    :param scalar: correction towards Pareto frontier, prevents weakly pareto dominant solution (~0.0001 to 0.01)
    :param nested_opt_method: optimization method to use
    :param nested_opt_args: arguments for the optimization method
    :return: an design point that is likely to result in objective values closest to the weight goal objective
    """
    wrapper_args = {"f": f,
                    "f_args": f_args,
                    "goal_obj": goal_obj,
                    "weights": weights,
                    "scalar": scalar}
    return optimization.optimize(min_max_wrapper, x_0, wrapper_args, nested_opt_method, **nested_opt_args)


def weighted_min_max_scan_method(f, x_0, f_args, goal_obj, weight_step=0.1, scalar=0.01,
                            nested_opt_method="newtons_method", nested_opt_args={}):
    """
    Varies a set of weights for each objective and optimizes the weighted infinite-norm distance from a goal objective

    :param f: objective function
    :param x_0: initial design point
    :param f_args: immutable dictionary of arguments for the objective function
    :param goal_obj: goal objective
    :param weight_step: step size between each weight array
    :param scalar: small correction that works to prevent weakly pareto dominant points (~0.0001 to 0.01)
    :param nested_opt_method: optimization method to use
    :param nested_opt_args: arguments for the optimization method
    :return: array of design points likely creating a Pareto frontier
    """
    scan_args = {"f": f,
                 "x_0": x_0,
                 "f_args": f_args,
                 "goal_obj": goal_obj,
                 "scalar": scalar,
                 "nested_opt_method": nested_opt_method,
                 "nested_opt_args": nested_opt_args}
    return scan_weights(weight_step, len(goal_obj), weighted_min_max_method, scan_args)


def exponential_weight_wrapper(x, f, f_args, weights, p):
    """
    Calculates the exponential weighted criterion for a design point

    :param x: design point
    :param f: objective function
    :param f_args: immutable dictionary of arguments for an objective functions
    :param weights: array of weights for each objective
    :param p: power to take exponentials for weights and objectives to
    :return: exponentially weighted function value
    """
    return sum((np.exp(p * weights) - 1) * np.exp(p * f(x, **f_args)))


def exponential_weight_method(f, x_0, f_args, weights, p=1,
                              nested_opt_method="newtons_method", nested_opt_args={}):
    """
    Optimizes the exponential weighted criterion for the objective function

    :param f: objective function
    :param x_0: initial design point
    :param f_args: immutable dictionary of arguments for the objective function
    :param weights: array of weights for each objective
    :param p: power to take exponentials to
    :param nested_opt_method: optimization method to use
    :param nested_opt_args: arguments for the optimization method
    :return: a likely optimal design point for given objective weights
    """
    args = {"f": f,
            "f_args": f_args,
            "weights": weights,
            "p": p}
    return optimization.optimize(exponential_weight_wrapper, x_0, args, nested_opt_method, **nested_opt_args)


def exponential_weight_scan_method(f, x_0, f_args, obj_count, weight_step=0.1, p=1,
                                   nested_opt_method="newtons_method", nested_opt_args={}):
    """
    Varies a set of weights for each objective and optimizes the exponential weighted criterion for an objective function

    :param f: objective function
    :param x_0: initial design point
    :param f_args: immutable dictionary of argument for the objective function
    :param obj_count: number of objectives in function output
    :param weight_step: step size between weight arrays
    :param p: power to take exponentials to
    :param nested_opt_method: optimization method to use
    :param nested_opt_args: arguments for the optimization method
    :return: an array of design points likely to make a Pareto frontier
    """
    scan_args = {"f": f,
                 "x_0": x_0,
                 "f_args": f_args,
                 "p": p,
                 "nested_opt_method": nested_opt_method,
                 "nested_opt_args": nested_opt_args}
    return scan_weights(weight_step, obj_count, exponential_weight_method, scan_args)


def non_domination_levels(f, X, f_args, multithreaded=False):
    vals = batch_eval(f, X, f_args, multithreaded)
    levels = []
    while len(X) > 0:
        this_level = []
        i = 0
        while i < len(X):
            if not any(dominates(vals[j], vals[i]) for j in range(len(X))):
                this_level.append(X.pop(i))
                vals.pop(i)
            else:
                i += 1
        levels.append(this_level)
    return levels
