from lineSearch import line_search
import numpy as np


def coordinate_descent(f, x_0, f_args, accel=False, lin_method="full_search", lin_args=None, max_steps=50, tol=0.00001):
    if lin_args is None:
        lin_args = {}
    x_curr = np.array(x_0)
    x_prev = np.ones(len(x_0)) * np.inf
    while max_steps > 0 and np.linalg.norm(x_curr - x_prev) > tol:
        x_prev = x_curr.copy()
        for i in range(len(x_0)):
            d = np.zeros(len(x_0))
            d[i] = 1
            x_curr = line_search(f, x_curr, f_args, d, lin_method, lin_args)
        if accel:
            diag = x_curr - np.array(x_0)
            x_curr = line_search(f, x_curr, f_args, diag / np.linalg.norm(diag), lin_method, lin_args)
        max_steps -= 1
    return x_curr


def powells_method(f, x_0, f_args, lin_method='full_search', lin_args=None, max_steps=50, tol=0.00001):
    n = len(x_0)
    U = np.identity(n)
    last = np.zeros(n)
    last[n-1] = 1
    reset = False

    x_curr = np.array(x_0)
    x_prev = np.ones(n) * np.inf
    while max_steps > 0 and np.linalg.norm(x_curr - x_prev) > tol:
        x_prev = x_curr.copy()

        for i in range(n):
            x_curr = line_search(f, x_curr, f_args, U[i], lin_method, lin_args)
            U[i] = U[(i+1) % n].copy()
        U[n-1] = x_prev - x_curr
        x_curr = line_search(f, x_curr, f_args, U[n-1], lin_method, lin_args)
        max_steps -= 1

        if reset:
            U = np.identity(n)
            reset = False
        elif all(U[0][i] == last[i] for i in range(n)):
            reset = True
    return x_curr


def hooke_jeeves_method(f, x_0, f_args, alpha=1, beta=0.5, max_steps=50, tol=0.001, multithreaded=False):    # TODO: Implement multithreading function
    n = len(x_0)
    D = []
    for i in range(n):
        a = np.zeros(n)
        a[i] = 1
        D.append(a.copy())
        D.append(-a)

    return pattern_search(f, x_0, f_args, D, alpha, beta, max_steps, tol, multithreaded)



def pattern_search(f, x_0, f_args, D=None, alpha=1, beta=0.5, max_steps=50, tol=0.001, multithreaded=False):
    n = len(x_0)
    x_curr = x_0.copy()

    if D is None:
        # TODO: Algorithmically determine a minimal positive spanning set, defaulting to Hooke-Jeeves now
        D = []
        for i in range(n):
            a = np.zeros(n)
            a[i] = 1
            D.append(a.copy())
            D.append(-a)

    f_min = f(x_curr, **f_args)
    x_min = x_curr.copy()
    while max_steps > 0 and alpha > tol:
        for d in D:
            x = x_curr + alpha * d
            c = f(x, **f_args)
            if c < f_min:
                f_min = c
                x_min = x
        if all(x_min[i] == x_curr[i] for i in range(n)):
            alpha *= beta
        else:
            x_curr = x_min.copy()
        max_steps -= 1
    return x_curr
