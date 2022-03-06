import numpy as np
from hessianGroup import HessianGroup as hg
from gradientGroup import GradientGroup as gg
import lineSearch as ls


def newtons_method(f, x_0, f_args, f_prime=None, fp_args=None, f_dprime=None, fdp_args=None, auto_diff=True,
                   max_steps=50, tol=0.0001, lin_method=None, lin_args=None):
    if fp_args is None:
        fp_args = {}
    if fdp_args is None:
        fdp_args = {}
    if lin_args is None:
        lin_args = {}
    x_curr = np.array(x_0)
    x_prev = np.ones(len(x_0)) * np.inf

    while max_steps > 0 and np.linalg.norm(x_curr - x_prev) > tol:
        x_prev = x_curr.copy()

        if auto_diff:
            temp = f(hg.make_hessian_groups(x_curr), **f_args)
            grad = temp.gradients
            hess = temp.hessian
        else:
            grad = f_prime(x_curr, **fp_args)
            hess = f_dprime(x_curr, **fdp_args)

        d = -np.matmul(np.linalg.inv(hess), grad).flatten()
        if lin_method is None:
            x_curr = x_prev + d
        else:
            x_curr = ls.line_search(f, x_curr, f_args, d, method=lin_method, lin_args=lin_args)
        max_steps -= 1

    return x_curr


def quasi_newtons_method(f, x_0, f_args, f_prime=None, fp_args=None, auto_diff=True, appr="dfp",
                         max_steps=50, tol=0.0001, lin_method="strong_backtrack_search", lin_args=None):
    if fp_args is None:
        fp_args = {}
    if lin_args is None:
        lin_args = {}
    x_curr = np.array(x_0)
    x_prev = np.ones(len(x_0)) * np.inf
    Q = np.identity(len(x_0))

    if auto_diff:
        grad = f(gg.make_gradient_groups(x_curr), **f_args).gradients
    else:
        grad = f_prime(x_curr, **fp_args)

    while max_steps > 0 and np.linalg.norm(x_curr - x_prev) > tol:
        x_prev = x_curr.copy()
        grad_prev = grad.copy()

        d = -np.matmul(Q, grad).flatten()
        x_curr = ls.line_search(f, x_curr, f_args, d, method=lin_method, lin_args=lin_args)

        if auto_diff:
            grad = f(gg.make_gradient_groups(x_curr), **f_args).gradients
        else:
            grad = f_prime(x_curr, **fp_args)

        gamma = grad - grad_prev
        gamma_f = gamma.flatten()
        delta = (x_curr - x_prev).reshape((len(x_0), 1))
        delta_f = delta.flatten()
        if appr == "dfp":
            Q -= np.matmul(np.matmul(np.matmul(Q, gamma), np.transpose(gamma)), Q) / \
                 np.dot(np.matmul(np.transpose(gamma), Q), gamma)
            Q += np.matmul(delta, np.transpose(delta)) / np.dot(delta_f, gamma)
        else:
            temp1 = (np.matmul(np.matmul(delta, np.transpose(gamma)), Q) + np.matmul(np.matmul(Q, gamma),
                     np.transpose(delta))) / np.dot(delta_f, gamma)
            temp2 = (1 + np.dot(np.matmul(np.transpose(gamma), Q), gamma) / np.dot(delta_f, gamma)) * \
                     np.matmul(delta, np.transpose(delta)) / np.dot(delta_f, gamma)
            Q += temp2 - temp1
        max_steps -= 1

    return x_curr




