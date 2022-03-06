import numpy as np


# This function returns 2 constants, ensuring a local min on f(x) is enclosed
def bracket_minimum(f, step=1, growth=2, max_steps=10):
    a = 0
    b = step
    c = 0
    ya, yb = f(a), f(b)
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


def golden_section_search(f, a, b, n=10):
    p = 0.61803    # golden ration - 1
    d = p * b + (1 - p) * a
    while n > 0:
        c = p * a + (1 - p) * b
        if f(c) < f(d):
            b = d
            d = c
        else:
            a = b
            b = c
        n -= 1
    return a, b


# a < b < c
def quadratic_fit_search(f, a, b, c=None, n=10):
    if c is None:
        c = b
        b = (a + c) / 2
    ya, yb, yc = f(a), f(b), f(c)
    while n > 0:
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
        n -= 1
    return a, c

