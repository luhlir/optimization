import numpy as np
import bracket
from gradientGroup import GradientGroup as gg


# Returns a new design point in the direction d from x
def line_search(f, x, f_args, d, method="strong_backtrack_search", lin_args=None):
    if len(x) != len(d):
        return x
    if lin_args is None:
        lin_args = {}

    if method == "full_search":
        a = full_search(f, x, f_args, d, **lin_args)
    elif method == "fix_step":
        a = fix_search(**lin_args)
    elif method == "decaying_step":
        a = decaying_search(**lin_args) # Make some better global constants to implement this?
    elif method == "backtrack_search":
        a = backtrack_search(f, x, f_args, d, **lin_args)
    elif method == "strong_backtrack_search":
        a = strong_backtrack_search(f, x, f_args, d, **lin_args)
    else:
        a = 0
    return x + a * d


# TODO: implement a full line search using bracketing methods
# Searches the 1D function of f(x + z*d) to find best step size of a
def full_search(f, x, f_args, d, brac_method="golden_section_search", brac_args=None, min_args=None):
    if brac_args is None:
        brac_args = {}
    if min_args is None:
        min_args = {}

    def f_wrap(a):
        return f(x + a * d, **f_args)

    b, c = bracket.bracket_minimum(f_wrap, **min_args)
    if brac_method == "golden_section_search":
        y, z = bracket.golden_section_search(f_wrap, b, c, **brac_args)
    elif brac_method == "quadratic_fit_search":
        y, z = bracket.quadratic_fit_search(f_wrap, b, c, **brac_args)
    else:
        y, z = b, c
    return (y + z) / 2


# Fixed step takes the same step every time
def fix_search(alpha):
    return alpha


global k


# Decaying step takes a smaller step every time
def decaying_search(alpha, decay):
    k += 1
    return alpha * (decay ** k)   # Better way to do this


# Backtracking search, find spot where function is at most what we expect from gradient
def backtrack_search(f, x, f_args, d, f_prime=None, fp_args=None, auto_diff=True, alpha=1, p=0.5, beta=0.0001,
                     max_steps=20):
    if p > 1:
        p = 1 / p
    if fp_args is None:
        fp_args = {}

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


# Strong backtracking using the strong Wolfe conditions
def strong_backtrack_search(f, x, f_args, d, f_prime=None, fp_args=None, auto_diff=True, alpha=1, p=0.5, beta=0.0001,
                            sig=0.1, max_steps=20):
    if p < 1:
        p = 1 / p
    if max_steps <= 0:
        return 0
    if fp_args is None:
        fp_args = {}

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
