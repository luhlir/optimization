import numpy as np
import optimization


def hyperrectangle_transform(x_hat, lower, upper):
    """
    Transforms a vector in R^n to a vector in the hyper rectangle [lower, upper]

    :param x_hat: input vector
    :param lower: lower bound for the hyper rectangle
    :param upper: upper bound for the hyper rectangle
    :return: vector in [lower, upper]
    """
    return (upper + lower) / 2 + (upper - lower) * x_hat / (1 + x_hat ** 2)


def hyperrectangle_function(x_hat, lower, upper, f, f_args):
    """
    Wrapper in R^n for a function in the hyper rectangle [lower, upper]

    :param x_hat: input vector in R^n
    :param lower: lower bound of the hyper rectangle
    :param upper: upper bound of the hyper rectangle
    :param f: function to evaluate
    :param f_args: dictionary of immutable arguments for the function
    :return: value of the function within the hyper rectangle
    """
    return f(hyperrectangle_transform(x_hat, lower, upper), **f_args)


def hyperrectangle_constraint(f, x_0, f_args, upper, lower, method="newtons_method", opt_args={}):
    """
    Constrains the optimization method to a hyper rectangle of [lower, upper]

    :param f: objective function
    :param x_0: starting design point
    :param f_args: dictionary of immutable arguments for the objective function
    :param upper: upper bound of the hyper rectangle
    :param lower: lower bound of the hyper rectangle
    :param method: optimization method to use
    :param opt_args: dictionary of immutable arguments for the optimization method
    :return: a likely locally optimal design point
    """
    transform_args = {"lower": np.array(lower),
                      "upper": np.array(upper),
                      "f": f,
                      "f_args": f_args}
    x_hat = optimization.optimize(hyperrectangle_function, x_0, transform_args, method, **opt_args)
    print(x_hat)
    return hyperrectangle_transform(x_hat, np.array(lower), np.array(upper))


def penalty_function(x, f, f_args, equality_constraints, inequality_constraints, penalty_type, scalar_1, scalar_2):
    """
    Evaluates a function and adds a penalty outside the limits described by the constraint functions

    :param x: vector to evaluate at
    :param f: function to evaluate
    :param f_args: dictionary of immutable function arguments
    :param equality_constraints: list of constraint functions such that h(x) = 0
    :param inequality_constraints: list of constraint functions such that g(x) <= 0
    :param penalty_type: "count", "quadratic", or "mixed" penalties used
    :param scalar_1: scalar to augment the "count" penalty for not satisfying a constraint function
    :param scalar_2: scalar to augment the "quadratic" penalty for not satisfying a constraint function
    :return: a penalized function value
    """
    penalty_1 = 0
    penalty_2 = 0
    if penalty_type == "count" or penalty_type == "mixed":
        for g in inequality_constraints:
            if g(x) > 0:
                penalty_1 += 1
        for h in equality_constraints:
            if h(x) != 0:
                penalty_1 += 1
    if penalty_type == "quadratic" or penalty_type == "mixed":
        for g in inequality_constraints:
            penalty_2 += max(g(x), 0) ** 2
        for h in equality_constraints:
            penalty_2 += h(x) ** 2
    return f(x, **f_args) + scalar_1 * penalty_1 + scalar_2 * penalty_2


def penalty_constraint(f, x_0, f_args, equality_constraints=[], inequality_constraints=[], penalty_type="mixed",
                       scalar_1=1, scalar_2=1, correction=2, method="newtons_method", opt_args={}, max_steps=10):
    """
    Optimizes an objective function using a penalty for not satisfying the given constraints

    :param f: objective function
    :param x_0: starting design point
    :param f_args: dictionary of immutable arguments for the objective function
    :param equality_constraints: list of constraint functions such that h(x) = 0
    :param inequality_constraints: list of constraint functions such that g(x) <= 0
    :param penalty_type: "count", "quadratic", or "mixed" penalties used
    :param scalar_1: scalar to augment the "count" penalty for not satisfying a constraint function
    :param scalar_2: scalar to augment the "quadratic" penalty for not satisfying a constraint function
    :param correction: scalar >1 to increase penalty in consecutive optimization calls
    :param method: optimization method to use
    :param opt_args: dictionary of immutable arguments for the optimization method
    :param max_steps: maximum number of calls to the optimization method without returning a design point within the constraints
    :return: a likely locally minimal design point within the constraints
    """
    x_curr = np.array(x_0)
    while max_steps > 0:
        penalty_args = {"f": f,
                        "f_args": f_args,
                        "equality_constraints": equality_constraints,
                        "inequality_constraints": inequality_constraints,
                        "penalty_type": penalty_type,
                        "scalar_1": scalar_1,
                        "scalar_2": scalar_2}
        x_curr = optimization.optimize(penalty_function, x_curr, penalty_args, method, **opt_args)
        error = 0
        for g in inequality_constraints:
            error += max(g(x_curr), 0)
        for h in equality_constraints:
            error += abs(h(x_curr))
        if error == 0:
            break
        scalar_1 *= correction
        scalar_2 *= correction
        max_steps -= 1
    return x_curr


def lagrange_function(x, f, f_args, equality_constraints, scalar, dynamic):
    """
    Adds a penalty to a given function according to the augmented Lagrange method

    :param x: point to evaluate at
    :param f: function to evaluate
    :param f_args: dictionary of immutable arguments for the function
    :param equality_constraints: list of constraint functions to satisfy h(x) = 0
    :param scalar: scalar to augment the static penalty value
    :param dynamic: dynamic array of scalars
    :return: the penalized function value
    """
    h = [constraint(x) for constraint in equality_constraints]
    penalty = 0
    for i in range(len(h)):
        penalty += scalar / 2 * (h[i] ** 2) - dynamic[i] * h[i]
    return f(x, **f_args) + penalty


def augmented_lagrange(f, x_0, f_args, equality_constraints=[], scalar=1, correction=2,
                       method="newtons_method", opt_args={}, max_steps=10):
    """
    Optimizes a function according to equality constraints using static and dynamic penalty scalars

    :param f: objective function
    :param x_0: starting design point
    :param f_args: dictionary of immutable arguments for the objective function
    :param equality_constraints: list of constraint functions such that h(x) = 0
    :param scalar: static penalty scalar to increase with each optimization call
    :param correction: scalar >1 to increase static penalty scalar by at each step
    :param method: optimization method to use
    :param opt_args: dictionary of immutable arguments for the optimization method
    :param max_steps: number of calls to the optimization algorithm
    :return: a likely locally minimal design point within the constraints
    """
    dynamic = np.zeros(len(equality_constraints))
    x_curr = np.array(x_0)
    while max_steps > 0:
        penalty_args = {"f": f,
                        "f_args": f_args,
                        "equality_constraints": equality_constraints,
                        "scalar": scalar,
                        "dynamic": dynamic}
        x_curr = optimization.optimize(lagrange_function, x_curr, penalty_args, method, **opt_args)
        for i in range(len(equality_constraints)):
            dynamic[i] -= scalar * equality_constraints[i](x_curr)
        scalar *= correction
        max_steps -= 1
    return x_curr


def barrier_function(x, f, f_args, inequality_constraints, penalty_type, scalar):
    """
    Adds a penalty to the function value near but not outside of the inequality constraint barrier

    :param x: point to evaluate function at
    :param f: function to evaluate
    :param f_args: dictionary of immutable arguments for the function
    :param inequality_constraints: list of constraint functions such that g(x) <= 0
    :param penalty_type: "inverse" or "log" to sum up penalties near barriers
    :param scalar: scalar to divide penalty by
    :return: the penalized function value
    """
    penalty = 0
    if penalty_type == "inverse":
        for g in inequality_constraints:
            penalty -= 1 / g(x)
    else:
        for g in inequality_constraints:
            offset = g(x)
            if offset >= -1:
                penalty -= np.log10(-offset)
    return f(x, **f_args) + penalty / scalar


def interior_point_method(f, x_0, f_args, inequality_constraints=[], scalar=1, correction=2, penalty_type="log",
                          method="newtons_method", opt_args={}, tol=0.001, max_steps=50):
    """
    Constrains optimization algorithm by penalizing function near inequality constraint barriers but not outside them

    :param f: objective function
    :param x_0: starting design point
    :param f_args: dictionary of immutable arguments for the objective function
    :param inequality_constraints: list of constraint functions such that g(x) <= 0
    :param scalar: scalar to divide penalty by
    :param correction: scalar to multiply penalty scalar by between consecutive calls to the optimization algorithm
    :param penalty_type: "inverse" or "log" methods of summing penalties
    :param method: optimization method to use
    :param opt_args: dictionary of immutable arguments for the optimization algorithm
    :param tol: necessary movement between calls to the optimization algorithm to continue with optimization
    :param max_steps: maximum number of calls to the optimization algorithm to make
    :return: a likely locally optimal design point within the inequality constraints
    """
    x_curr = np.array(x_0)
    delta = np.inf
    while delta > tol and max_steps > 0:
        penalty_args = {"f": f,
                        "f_args": f_args,
                        "inequality_constraints": inequality_constraints,
                        "penalty_type": penalty_type,
                        "scalar": scalar}
        x_new = optimization.optimize(barrier_function, x_curr, penalty_args, method, **opt_args)
        delta = np.linalg.norm(x_curr - x_new)
        x_curr = x_new.copy()
        scalar *= correction
        max_steps -= 1
    return x_curr
