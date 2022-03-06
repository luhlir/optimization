import numpy as np
from gradientGroup import GradientGroup as gg
import lineSearch as ls


def gradient_descent(f, x_0, f_args, f_prime=None, fp_args=None, auto_diff=True, max_steps=50,
                     tol=0.00001, lin_method="strong_backtrack_search", lin_args=None):
    if fp_args is None:
        fp_args = {}
    if lin_args is None:
        lin_args = {}
    x_curr = np.array(x_0)
    x_prev = np.ones(len(x_0)) * np.inf

    while max_steps > 0 and np.linalg.norm(x_curr - x_prev) > tol:
        x_prev = x_curr.copy()
        max_steps -= 1

        if auto_diff:
            grad = f(gg.make_gradient_groups(x_curr), **f_args).gradients.flatten()
        else:
            grad = f_prime(x_curr, **fp_args)
        d = -grad / np.linalg.norm(grad)
        x_curr = ls.line_search(f, x_curr, f_args, d, method=lin_method, lin_args=lin_args)
    return x_curr


def conjugate_descent(f, x_0, f_args, f_prime=None, fp_args=None, auto_diff=True, beta_method="polak-ribiere",
                      max_steps=50, tol=0.00001, lin_method="strong_backtrack_search", lin_args=None):
    if fp_args is None:
        fp_args = {}
    if lin_args is None:
        lin_args = {}
    x_curr = np.array(x_0)
    x_prev = np.ones(len(x_0)) * np.inf
    d_prev = np.zeros(len(x_0))
    g_prev = np.ones(len(x_0))

    while max_steps > 0 and np.linalg.norm(x_curr - x_prev) > tol:
        x_prev = x_curr.copy()
        max_steps -= 1

        if auto_diff:
            grad = f(gg.make_gradient_groups(x_curr), **f_args).gradients.flatten()
        else:
            grad = f_prime(x_curr, **fp_args)

        if beta_method == "polak-ribiere":
            beta = max(np.dot(grad, grad - g_prev) / np.dot(g_prev, g_prev), 0)
        else:   # Fletcher-Reeves
            beta = np.dot(grad, grad) / np.dot(g_prev, g_prev)
        d = -grad + beta * d_prev

        d_prev = d.copy()
        g_prev = grad.copy()
        x_curr = ls.line_search(f, x_curr, f_args, d, method=lin_method, lin_args=lin_args)
    return x_curr


def momentum_descent(f, x_0, f_args, f_prime=None, fp_args=None, auto_diff=True, alpha=0.01, beta=0.01,
                     max_steps=50, tol=0.00001):
    if fp_args is None:
        fp_args = {}
    v = np.zeros(len(x_0))
    x_prev = np.ones(len(x_0)) * np.inf
    x_curr = np.array(x_0)

    while max_steps > 0 and np.linalg.norm(x_curr - x_prev) > tol:
        x_prev = x_curr.copy()

        if auto_diff:
            grad = f(gg.make_gradient_groups(x_curr), **f_args).gradients.flatten()
        else:
            grad = f_prime(x_curr, **fp_args)

        v = beta * v - alpha * grad
        x_curr = x_prev + v

        max_steps -= 1

    return x_curr


def nesterov_descent(f, x_0, f_args, f_prime=None, fp_args=None, auto_diff=True, alpha=0.01, beta=0.01,
                     max_steps=50, tol=0.00001):
    if fp_args is None:
        fp_args = {}
    v = np.zeros(len(x_0))
    x_prev = np.ones(len(x_0)) * np.inf
    x_curr = np.array(x_0)

    while max_steps > 0 and np.linalg.norm(x_curr - x_prev) > tol:
        x_prev = x_curr.copy()

        if auto_diff:
            grad = f(gg.make_gradient_groups(x_curr + beta * v), **f_args).gradients.flatten()
        else:
            grad = f_prime(x_curr + beta * v, **fp_args)

        v = beta * v - alpha * grad
        x_curr = x_prev + v

        max_steps -= 1

    return x_curr


def adagrad_descent(f, x_0, f_args, f_prime=None, fp_args=None, auto_diff=True, alpha=0.01, max_steps=50, tol=0.00001):
    if fp_args is None:
        fp_args = {}
    s = np.zeros(len(x_0))
    x_prev = np.ones(len(x_0)) * np.inf
    x_curr = np.array(x_0)

    while max_steps > 0 and np.linalg.norm(x_curr - x_prev) > tol:
        x_prev = x_curr.copy()

        if auto_diff:
            grad = f(gg.make_gradient_groups(x_curr), **f_args).gradients.flatten()
        else:
            grad = f_prime(x_curr, **fp_args)

        s += grad * grad
        x_curr = x_prev - (alpha / (10 ** -8 + np.sqrt(s))) * grad

        max_steps -= 1

    return x_curr


def rmsprop_descent(f, x_0, f_args, f_prime=None, fp_args=None, auto_diff=True, alpha=0.01, gamma=0.9,
                    max_steps=50, tol=0.00001):
    if fp_args is None:
        fp_args = {}
    s = np.zeros(len(x_0))
    x_prev = np.ones(len(x_0)) * np.inf
    x_curr = np.array(x_0)

    while max_steps > 0 and np.linalg.norm(x_curr - x_prev) > tol:
        x_prev = x_curr.copy()

        if auto_diff:
            grad = f(gg.make_gradient_groups(x_curr), **f_args).gradients.flatten()
        else:
            grad = f_prime(x_curr, **fp_args)

        s = gamma * s + (1 - gamma) * (grad * grad)
        x_curr = x_prev - (alpha / (10 ** -8 + np.sqrt(s))) * grad

        max_steps -= 1

    return x_curr


def adadelta_descent(f, x_0, f_args, f_prime=None, fp_args=None, auto_diff=True, delta=0.9, gamma=0.9,
                     max_steps=50, tol=0.00001):
    if fp_args is None:
        fp_args = {}
    s = np.zeros(len(x_0))
    dx = np.zeros(len(x_0))
    x_prev = np.ones(len(x_0)) * np.inf
    x_curr = np.array(x_0)
    e = 10 ** -8

    while max_steps > 0 and np.linalg.norm(x_curr - x_prev) > tol:
        x_prev = x_curr.copy()

        if auto_diff:
            grad = f(gg.make_gradient_groups(x_curr), **f_args).gradients.flatten()
        else:
            grad = f_prime(x_curr, **fp_args)

        s = gamma * s + (1 - gamma) * (grad * grad)
        update = - ((e + np.sqrt(dx)) / (e + np.sqrt(s))) * grad
        dx = delta * dx + (1 - delta) * (update * update)
        x_curr = x_prev + dx

        max_steps -= 1

    return x_curr


def adam_descent(f, x_0, f_args, f_prime=None, fp_args=None, auto_diff=True, alpha=0.001, delta=0.9, gamma=0.999,
                 max_steps=50, tol=0.00001):
    if fp_args is None:
        fp_args = {}
    s = np.zeros(len(x_0))
    v = np.zeros(len(x_0))
    x_prev = np.ones(len(x_0)) * np.inf
    x_curr = np.array(x_0)
    e = 10 ** -8
    k = 1

    while max_steps > k and np.linalg.norm(x_curr - x_prev) > tol:
        x_prev = x_curr.copy()

        if auto_diff:
            grad = f(gg.make_gradient_groups(x_curr), **f_args).gradients.flatten()
        else:
            grad = f_prime(x_curr, **fp_args)

        v = delta * v + (1 - delta) * grad
        s = gamma * s + (1 - gamma) * grad * grad
        dv = v / (1 - (delta ** k))
        ds = s / (1 - (gamma ** k))
        x_curr = x_prev - alpha * dv / (e + np.sqrt(ds))

        k += 1

    return x_curr


def hypergradient_descent(f, x_0, f_args, f_prime=None, fp_args=None, auto_diff=True, alpha=0.1, mu=0.001,
                          max_steps=50, tol=0.00001):
    x_curr = np.array(x_0)
    x_prev = np.ones(len(x_0)) * np.inf
    grad_prev = np.zeros(len(x_0))

    while max_steps > 0 and np.linalg.norm(x_curr - x_prev) > tol:
        x_prev = x_curr.copy()

        if auto_diff:
            grad = f(gg.make_gradient_groups(x_curr), **f_args).gradients.flatten()
        else:
            grad = f_prime(x_curr, **fp_args)

        alpha += mu * np.dot(grad, grad_prev)
        x_curr = x_prev - alpha * grad

        grad_prev = grad.copy()
        max_steps -= 1
    return x_curr
