from enum import Enum

import numpy as np
from bracket import Bracket, bracket_minimum
from gradientGroup import GradientGroup as gg


def full_search(f, x, f_args, d, brac_method=Bracket.GOLDEN_SECTION_SEARCH, brac_args={}, min_args={}):
    """
    Performs a full search using a bracketing method to detect a local minimum on the given line

    :param f: objective function to search
    :param x: initial design point
    :param f_args: dictionary of immutable arguments for the objective function
    :param d: direction of line to search
    :param brac_method: bracketing method to use in the search
    :param brac_args: dictionary of immutable arguments for the bracketing method
    :param min_args: dictionary of immutable arguments for determining a good initial guess
    :return: a scalar describing the step size to take in the given direction
    """
    def f_wrap(a):
        return f(x + a * d, **f_args)

    b, c = bracket_minimum(f_wrap, **min_args)
    match brac_method:
        case Bracket.BRACKET_MINIMUM:
            y, z = b, c
        case _:
            y, z = brac_method.__call__(f_wrap, b, c, **brac_args)
    return (y + z) / 2


def fix_search(f, x, f_args, d, alpha):
    """
    Fixed step size

    :param f: unused
    :param x: unused
    :param f_args: unused
    :param d: unused
    :param alpha: the step size to take
    :return: the step size to take
    """
    return alpha


global k


def decaying_search(f, x, f_args, d, alpha, decay):
    """
    Decreases the step distance each time it's called

    :param f: unused
    :param x: unused
    :param f_args: unused
    :param d: unused
    :param alpha: initial step size
    :param decay: multiplicative decay each step
    :return: the step size to take this time
    """
    k += 1
    return alpha * (decay ** k)   # TODO: Better way to do this? <--- pass in the step size as an argument. use dictionary in upper levels


def backtrack_search(f, x, f_args, d, f_prime=None, fp_args={}, auto_diff=True, alpha=1, p=0.5, beta=0.0001,
                     max_steps=20):
    """
    Uses function values to find point that is likely to be minimal on the line based on the gradient

    :param f: function to evaluate
    :param x: starting point for search
    :param f_args: dictionary of immutable arguments to f
    :param d: direction vector to search in
    :param f_prime: gradient method of f
    :param fp_args: dictionary of immutable arguments to f_prime
    :param auto_diff: use automatic differentiation instead of an explicit gradient method
    :param alpha: initial step size
    :param p: step size correction factor <1 used at each step
    :param beta: scalar for expected change in function value
    :param max_steps: maximum number of function evaluations to take
    :return: scalar describing the step size to take in the given direction
    """
    if p > 1:
        p = 1 / p

    if auto_diff:
        res = f(gg.make_gradient_groups(x), **f_args)
        value = res.val
        gradient = res.gradients.flatten()
    else:
        value = f(x, **f_args)
        gradient = f_prime(x, **fp_args)
    point_expected = np.dot(gradient, d)

    while f(x + alpha * d, **f_args) > value + beta * alpha * point_expected and max_steps > 0:
        alpha *= p
        max_steps -= 1

    return alpha


def strong_backtrack_search(f, x, f_args, d, f_prime=None, fp_args={}, auto_diff=True, alpha=1, p=0.5, beta=0.0001,
                            sig=0.1, max_steps=20):
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
    if p < 1:
        p = 1 / p
    if max_steps <= 0:
        return 0

    if auto_diff:
        res = f(gg.make_gradient_groups(x), **f_args)
        point_value = res.val
        point_gradient = res.gradients.flatten()
    else:
        point_value = f(x, **f_args)
        point_gradient = np.array(f_prime(x, **fp_args))
    point_expected = np.dot(point_gradient, d)

    last_alpha = 0
    while max_steps > 0:
        next_step = x + alpha * d
        if auto_diff:
            res = f(gg.make_gradient_groups(next_step), **f_args)
            next_value = res.val
            next_gradient = res.gradients.flatten()
        else:
            next_value = f(next_step, **f_args)
            next_gradient = f_prime(next_step, **fp_args)
        next_expected = np.dot(next_gradient, d)

        if next_value >= point_value or next_value > point_value + beta * alpha * point_expected or next_expected >= 0:
            break
        else:
            last_alpha = alpha
            alpha *= p
        max_steps -= 1

    # We've bracketed the wolfe condition zone to [last_alpha, alpha]
    # Check Wolfe conditions for the endpoint alpha
    if next_value <= point_value + beta * alpha * point_expected and abs(next_expected) <= -sig * point_expected:
        return alpha

    if auto_diff:
        res = f(gg.make_gradient_groups(x + last_alpha * d), **f_args)
        next_value = res.val
        next_gradient = res.gradients.flatten()
    else:
        next_value = f(next_step, **f_args)
        next_gradient = f_prime(next_step, **fp_args)
    next_expected = np.dot(next_gradient, d)

    # Check Wolfe conditions for the  endpoint last_alpha
    if next_value <= point_value + beta * last_alpha * point_expected and abs(next_expected) <= -sig * point_expected:
        return last_alpha

    midpoint = (alpha + last_alpha) / 2
    while max_steps > 0:
        midpoint = (alpha + last_alpha) / 2
        mid_step = x + midpoint * d
        if auto_diff:
            res = f(gg.make_gradient_groups(mid_step), **f_args)
            mid_value = res.val
            mid_gradient = res.gradients.flatten()
        else:
            mid_value = f(mid_step, **f_args)
            mid_gradient = f_prime(mid_step, **fp_args)
        mid_expected = np.dot(mid_gradient, d)
        # Check Wolfe conditions for midpoint and return if met
        if mid_value <= point_value + beta * midpoint * point_expected and abs(mid_expected) <= -sig * point_expected:
            return midpoint

        # Check boundary conditions for midpoint
        if mid_value >= point_value or mid_value > point_value + beta * midpoint * point_expected or mid_expected >= 0:
            alpha = midpoint
        else:
            last_alpha = midpoint
        max_steps -= 1
    return midpoint


class Search(Enum):
    FULL_SEARCH = full_search
    """
    Performs a full search using a bracketing method to detect a local minimum on the given line. Takes the following arguments:
    \nbrac_method - bracketing method to use in the search
    \nbrac_args - dictionary of immutable arguments for the bracketing method
    \nmin_args - dictionary of immutable arguments for determining a good initial guess
    """
    FIX_SEARCH = fix_search
    """
    Fixed step size. Takes the following arguments:
    \nalpha - the step size to take
    """
    DECAYING_SEARCH = decaying_search
    """
    Decreases the step distance each time it's called. Takes the following arguments:
    \nalpha - initial step size
    \ndecay - multiplicative decay each step
    """
    BACKTRACK_SEARCH = backtrack_search
    """
    Uses function values to find point that is likely to be minimal on the line based on the gradient. Takes the following arguments:
    \nf_prime - gradient method of f
    \nfp_args - dictionary of immutable arguments to f_prime
    \nauto_diff - use automatic differentiation instead of an explicit gradient method
    \nalpha - initial step size
    \np - step size correction factor <1 used at each step
    \nbeta - scalar for expected change in function value
    \nmax_steps - maximum number of function evaluations to take
    """
    STRONG_BACKTRACK_SEARCH = strong_backtrack_search
    """
    Performs a backtrack search that uses the strong Wolfe conditions to detect a local minimum on the line. Takes the following arguments:
    \nf_prime - gradient method of f
    \nfp_args - dictionary of immutable arguments to f_prime
    \nauto_diff - use automatic differentiation instead of an explicit gradient method
    \nalpha - initial step size
    \np - step size correction factor <1 used at each step in bracketing phase
    \nbeta - scalar for expected change in function value
    \nsig - scalar used in zoom phase
    \nmax_steps - maximum number of function calls to make
    """


def line_search(f, x, f_args, d, method=Search.STRONG_BACKTRACK_SEARCH, lin_args={}):
    """
    Performs the input line search method

    :param f: objective function to search
    :param x: design point to search around
    :param f_args: dictionary of immutable arguments for the objective function
    :param d: direction to search in
    :param method: line search method
    :param lin_args: dictionary of immutable arguments for the line search method
    :return: a good candidate design point satisfying x' = x + a * d
    """
    if len(x) != len(d):
        return x

    a = method.__call__(f, x, f_args, d, **lin_args)
    return x + a * d
