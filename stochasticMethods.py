import numpy as np
import scipy.stats as sp
import scipy.linalg as sp_lin
from sklearn.mixture import GaussianMixture
from gradientGroup import GradientGroup as gg
from zeroOrderMethods import make_minimal_positive_spanning_set, batch_eval


def noisy_descent(f, x_0, f_args, step_func, f_prime=None, fp_args=None, auto_diff=True,
                  max_steps=50, tol=None):
    """
    Modified gradient descent with gaussian noise.
    It's preferable to design the step size so the sum of all step sizes is infinity and the sum of their squares is
    less than infinity

    :param f: objective function
    :param x_0: starting design point
    :param f_args: dictionary of immutable arguments for the objective function
    :param step_func: takes current point, descent direction, and step size; returns a tuple of step length and desired noise standard deviation
    :param f_prime: gradient function for the objective function
    :param fp_args: dictionary of immutable arguments for the gradient function
    :param auto_diff: True if automatic differentiation will be used instead of a gradient function
    :param max_steps: maximum steps to take during optimization
    :param tol: tolerance, optimization terminates if the step size gets below this threshold
    :return: a likely locally minimal design point
    """
    if fp_args is None:
        fp_args = {}
    n = len(x_0)
    x_curr = np.array(x_0)
    x_prev = np.ones(len(x_0)) * np.inf

    for k in range(max_steps):
        if tol is not None and np.linalg.norm(x_curr - x_prev) < tol:
            break
        x_prev = x_curr.copy()

        if auto_diff:
            grad = f(gg.make_gradient_groups(x_curr), **f_args).gradients.flatten()
        else:
            grad = f_prime(x_curr, **fp_args)
        d = -grad / np.linalg.norm(grad)
        a, sigma = step_func(x_curr, d, k)
        x_curr = x_curr + a * d + sp.norm.rvs(scale=sigma, size=n)
    return x_curr


def mesh_adaptive_search(f, x_0, f_args, max_steps=100, tol=0.001):
    """
    Opportunistic pattern search using random minimal positive spanning set at each step

    :param f: objective function
    :param x_0: starting design point
    :param f_args: dictionary of immutable arguments for the objective function
    :param max_steps: maximum steps to take during optimization
    :param tol: tolerance, optimization terminates if the step size gets below this threshold
    :return: a likely locally minimal design point
    """
    n = len(x_0)
    x_curr = np.array(x_0)

    a, y_curr = 1, f(x_curr, **f_args)
    while max_steps > 0 and a > tol:
        stepped = False
        D = make_minimal_positive_spanning_set(a, n)
        for d in D:
            x_next = x_curr + a * d
            y_next = f(x_next, **f_args)
            if y_next < y_curr:
                x_curr, y_curr, stepped = x_next.copy(), y_next, True
                x_next = x_curr + 3 * a * d
                y_next = f(x_next, **f_args)
                if y_next < y_curr:
                    x_curr, y_curr = x_next.copy(), y_next
                break
        if stepped:
            a = min(4 * a, 1)
        else:
            a /= 4
        max_steps -= 1
    return x_curr


def simulated_annealing(f, x_0, f_args, K=None, t=None, max_steps=50):
    """
    At each step, accepts a better point or a worse point with decreasing probability
    Fixed transition point distribution

    :param f: objective function
    :param x_0: starting design point
    :param f_args: dictionary of immutable arguments for the objective function
    :param K: covariance matrix for the optimization space, default to identity matrix
    :param t: function that takes the step count as input, decreasing, defaults to 1/steps
    :param max_steps: maximum steps to take during optimization
    :return: a likely locally minimal design point
    """
    n = len(x_0)
    if K is None:
        K = np.identity(n)
    if t is None:
        def t(step):
            return 1 / step

    x_curr = np.array(x_0)
    x_best = x_curr.copy()
    y_curr = f(x_curr, **f_args)
    y_best = y_curr
    for k in range(max_steps):
        x = x_curr + np.random.multivariate_normal(np.zeros(n), K)
        y = f(x, **f_args)
        dy = y - y_curr
        if dy <= 0 or np.random.rand() < np.exp(-dy / t(k+1)):
            x_curr, y_curr = x.copy(), y
        if y < y_best:
            x_best, y_best = x.copy(), y
    return x_best


def corana_annealing(f, x_0, f_args, t=None, step_size_0=None, steps_per_update=10, step_variation=None, max_steps=50):
    """
    At each step, accepts a better point or a worse point with decreasing probability
    Dynamic transition point distribution based on acceptance rate in each dimension

    :param f: objective function
    :param x_0: starting design point
    :param f_args: dictionary of immutable arguments for the objective function
    :param t: function that takes the step count as input, decreasing, defaults to 1/steps
    :param step_size_0: starting step size, defaults to 1's
    :param steps_per_update: number of steps searched before updating step size
    :param step_variation: update parameter for how much step size changes each update
    :param max_steps: maximum steps to take during optimization
    :return: a likely locally minimal design point
    """
    n = len(x_0)
    if step_size_0 is None:
        step_size = np.ones(n)
    else:
        step_size = np.array(step_size_0)
    if step_variation is None:
        step_variation = np.ones(n) * 2
    if t is None:
        def t(step):
            # Reduce temp by half every 5 steps
            return 1 / step

    def corana_update(v, a, c, s):
        for i in range(n):
            if a[i] > 0.6 * s:
                v[i] *= 1 + (c[i] * (a[i] / s - 0.6) / 0.4)
            elif a[i] < 0.4 * s:
                v[i] /= 1 + (c[i] * (0.4 - a[i] / s) / 0.4)
        return v

    accepted = np.zeros(n)
    x_curr = np.array(x_0)
    x_best = x_curr.copy()
    y_curr = f(x_curr, **f_args)
    y_best = y_curr
    for k in range(max_steps):
        # Move in each direction randomly
        temp = t(k+1)
        for i in range(n):
            d = np.zeros(n)
            d[i] = 1
            x = x_curr + (2 * np.random.random() - 1) * step_size[i] * d
            y = f(x, **f_args)
            dy = y - y_curr
            if dy <= 0 or np.random.rand() < np.exp(-dy / temp):
                accepted[i] += 1
                x_curr, y_curr = x.copy(), y
            if y < y_best:
                x_best, y_best = x.copy(), y
        # Update the v vector every s steps
        if k % steps_per_update == 0:
            step_size = corana_update(step_size, accepted, step_variation, steps_per_update)
    # TODO: Update this based on the algorithm on page 134??? They seem to overcomplicate temperature and termination
    return x_best


def cross_entropy(f, f_args, P=None, P_init_args={}, x_0=None, sample_count=100, elite_count=10, multithreaded=False, max_steps=50):
    """
    Updates and uses a probability distribution object to explore the space

    :param f: objective function
    :param f_args: dictionary of immutable arguments for the objective function
    :param P: an object that implements fit(samples) and sample(sample_count) defaults to a scikit-learn GaussianMixture with 2 components
    :param P_init_args: dictionary of immutable arguments for P initialization at each step
    :param x_0: starting design point, if None, P is assumed to already be fitted
    :param sample_count: number of sample points to explore at each step
    :param elite_count: number of best sample points to fit on at each step
    :param multithreaded: determines if the samples will be processed in parallel or serial
    :param max_steps: maximum steps to take during optimization
    :return: a likely locally minimal design point
    """
    if P is None:
        # Default to GaussianMixture with 2 components
        if x_0 is None:
            raise Exception("Either an initial design point or a valid probability object must be given")
        P_init_args = {"n_components": 2}
        P = GaussianMixture(**P_init_args)

    x_best = np.array(x_0)
    y_best = np.inf

    if x_0 is not None:
        # If an initial guess is given, use it to initialize the probability distribution
        X = np.random.multivariate_normal(np.array(x_best), np.identity(len(x_0)), sample_count)
        Y = batch_eval(f, X, f_args, multithreaded)
        if np.min(Y) < y_best:
            x_best = X[np.argmin(Y)].copy()
            y_best = np.min(Y)
        indices = np.argsort(Y)
        P.fit(X[indices[:elite_count]])
        max_steps -= 1

    while max_steps > 0:
        X, _ = P.sample(sample_count)
        Y = batch_eval(f, X, f_args, multithreaded)
        if np.min(Y) < y_best:
            x_best = X[np.argmin(Y)].copy()
            y_best = np.min(Y)
        indices = np.argsort(Y)
        P = type(P)(**P_init_args)
        P.fit(X[indices[:elite_count]])
        max_steps -= 1
    return x_best


def natural_evolution(f, f_args, dist_params, get_samples, grad_log_likelihood, sample_size=100, step_size=0.001,
                      max_steps=50):
    """
    Uses gradient descent on a list of probability distribution parameters using the gradient of the log likelihood

    :param f: objective function
    :param f_args: dictionary of immutable arguments for the objective function
    :param dist_params: list of initial distribution parameters
    :param get_samples: function that takes the list of distribution parameters and number of samples to generate, returns list of samples
    :param grad_log_likelihood: function that takes a sample and list of distribution parameters, returns list of gradients of each parameter
    :param sample_size: number of samples to generate at each step
    :param step_size: scalar to multiply gradient at each step
    :param max_steps: maximum steps to take during optimization
    :return: a likely locally minimal design point
    """
    x_best = None
    y_best = np.inf
    while max_steps > 0:
        samples = get_samples(dist_params, sample_size)
        for sample in samples:
            y_curr = f(sample, **f_args)
            if y_curr < y_best:
                x_best = sample.copy()
                y_best = y_curr
            dist_update = grad_log_likelihood(sample, dist_params)
            for i in range(len(dist_params)):
                dist_params[i] -= step_size * y_curr * dist_update[i] / sample_size
        max_steps -= 1
    return x_best


def covariance_matrix_adaptation(f, x_0, f_args, step_size=1, sample_count=None, elite_count=None, multithreaded=False,
                                 max_steps=50):
    """
    Adapts a covariance matrix to

    :param f: objective function
    :param x_0: starting design point
    :param f_args: dictionary of immutable arguments for the objective function
    :param step_size: initial covariance scalar used in generating samples
    :param sample_count: number of samples to consider at each step
    :param elite_count: number of samples to base the new mean off of
    :param multithreaded: determines if the samples will be processed in parallel or serial
    :param max_steps: maximum number of steps to take
    :return: a likely locally minimal design point
    """
    n = len(x_0)
    if sample_count is None:
        sample_count = int(4 + np.floor(3 * np.log(n)))
        elite_count = int(np.floor(sample_count / 2))
    elif elite_count is None:
        elite_count = int(np.floor(sample_count / 2))
    weights = np.zeros(sample_count)
    for i in range(sample_count):
        weights[i] = np.log((sample_count + 1) / 2) - np.log(i + 1)
    weights[:int(np.floor(sample_count / 2))] /= sum(weights[:int(np.floor(sample_count / 2))])
    weights[int(np.floor(sample_count / 2)):] /= -sum(weights[int(np.floor(sample_count / 2)):])
    sel_mass = 1 / sum(weights[:elite_count] * weights[:elite_count])
    exp_dist = np.sqrt(n) * (1 - (1 / (4 * n)) + (1 / (21 * n * n)))
    c_step = (sel_mass + 2) / (n + sel_mass + 5)
    d_step = 1 + 2 * max(0, np.sqrt((sel_mass - 1) / (n + 1)) - 1) + c_step
    c_cov = (4 + (sel_mass / n)) / (n + 4 + (2 * sel_mass / n))
    c_1 = 2 / ((n + 1.3) ** 2 + sel_mass)
    c_mean = min(1 - c_1, 2 * (sel_mass - 2 + (1 / sel_mass)) / ((n + 2) ** 2 + sel_mass))

    p_step = 0
    p_cov = np.zeros((n, 1))
    mean = np.array(x_0)
    cov = np.identity(n)
    for k in range(max_steps):
        X = np.random.multivariate_normal(mean, step_size ** 2 * cov, sample_count)
        Y = batch_eval(f, X, f_args, multithreaded)
        indices = np.argsort(Y)
        X = X[indices]
        mean = np.zeros(n)
        for i in range(elite_count):
            mean += weights[i] * X[i]
        delta = X.copy()
        delta_w = np.zeros(n)
        for i in range(sample_count):
            delta[i] -= mean
            delta[i] /= step_size
            if i < elite_count:
                delta_w += weights[i] * delta[i]
        delta_w = np.reshape(delta_w, (n, 1))
        inv_sqrt_cov = sp_lin.sqrtm(sp_lin.inv(cov))

        p_cov *= (1 - c_cov)
        if np.linalg.norm(p_cov) / np.sqrt(1 - (1 - c_step) ** (2 * (k + 1))) < (1.4 + 2 / (n + 1)) * exp_dist:
            p_cov += np.sqrt(c_cov * (2 - c_step) * sel_mass) * delta_w
        else:
            cov *= 1 + c_1 * c_cov * (2 - c_cov) - c_1 - c_mean

        p_step = (1 - c_step) * p_step + np.sqrt(c_step * (2 - c_step) * sel_mass) * (inv_sqrt_cov * delta_w)[0][0]
        step_size *= np.exp((c_step / d_step) * ((abs(p_step) / exp_dist) - 1))

        cov += c_1 * p_cov * p_cov.transpose()
        for i in range(sample_count):
            delta_temp = np.reshape(delta[i], (n, 1))
            new_thing = c_mean * delta_temp * delta_temp.transpose()
            if weights[i] > 0:
                new_thing *= weights[i]
            elif weights[i] < 0:
                new_thing *= n * weights[i] / (np.linalg.norm(inv_sqrt_cov * delta_temp) ** 2)
            else:
                continue
            cov += new_thing
    return mean
