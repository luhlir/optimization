import numpy as np


def bracket_minimum(f, step=1, growth=2, max_steps=20):
    """Guarantees a local min of f is between the returned values. Does not guarantee a small bracket.

    :param f: Function takes a constant input and returns a constant output
    :param step: Initial step size
    :param growth: Growth factor by which the step increases
    :param max_steps: Maximum number of function calls, in case no local min exists (>2)
    :return: Tuple with 2 values that bracket a local min of f
    """
    a = 0
    b = step
    c = 0
    ya, yb = f(a), f(b)
    max_steps -= 2
    if ya < yb:
        temp = a
        a = b
        b = temp
        yb = ya
        step = -step
    while max_steps > 0:
        c = b + step
        yc = f(c)
        if yc > yb:
            break
        a = b
        b = c
        yb = yc
        step *= growth
        max_steps -= 1
    return a, c


def golden_section_search(f, lower, upper, steps=10):
    """Uses the golden ration to shrink the bracket at each step

    :param f: Function that takes a constant input and returns a constant output
    :param lower: Constant lower bound of bracket
    :param upper: Constant upper bound of bracket
    :param steps: Number of function calls (>1)
    :return: Tuple with 2 values that bracket a local min of f
    """
    p = 0.61803    # golden ration - 1
    d = p * upper + (1 - p) * lower
    yd = f(d)
    steps -= 1
    while steps > 0:
        c = p * lower + (1 - p) * upper
        yc = f(c)
        if yc < yd:
            upper = d
            d = c
            yd = yc
        else:
            lower = upper
            upper = c
        steps -= 1
    return lower, upper


def quadratic_fit_search(f, a, b, c=None, steps=10):
    """Interpolates a quadratic function around a, b, and c and brackets that min.

    a < b (< c if it is not None)

    :param f: Function that takes a constant input and returns a constant output
    :param a: Point around local min
    :param b: Point around local min (> a)
    :param c: Point around local min (> b), defaults to midpoint of a and b
    :param steps: Number of function calls (>3)
    :return: Tuple with 2 values that bracket a local min of f
    """
    if c is None:
        c = b
        b = (a + c) / 2
    ya, yb, yc = f(a), f(b), f(c)
    steps -= 3
    while steps > 0:
        x = ya * (b ** 2 - c ** 2) + yb * (c ** 2 - a ** 2) + yc * (a ** 2 - b ** 2)
        x /= ya * (b - c) + yb * (c - a) + yc * (a - b)
        x /= 2
        yx = f(x)
        if x > b:
            if yx > yb:
                c, yc = x, yx
            else:
                a, ya, b, yb = b, yb, x, yx
        elif x < b:
            if yx > yb:
                a, ya = x, yx
            else:
                c, yc, b, yb = b, yb, x, yx
        else:
            break
        steps -= 1
    return a, c

