import numpy as np
import scipy.stats as sp_stats
from zeroOrderMethods import batch_eval


def random_uniform(samples, lower, upper):
    """
    Generates an array of random samples within a bounded space

    :param samples: number of samples to generate
    :param lower: vector of lower bounds
    :param upper: vector of upper bounds
    :return: array of samples
    """
    n = len(lower)
    return [lower + np.random.random(n) * (upper - lower) for i in range(samples)]


def random_normal(samples, mean, covariance=None):
    """
    Generates an array of random samples normally distributed around a point

    :param samples: number of samples to generate
    :param mean: point to generate around
    :param covariance: covariance matrix to generate with
    :return: array of samples
    """
    n = len(mean)
    if covariance is None:
        covariance = np.identity(n)
    return np.random.multivariate_normal(mean, covariance, samples)


def random_cauchy(samples, mean, variance=None):
    """
    Generates an array of random samples using a cauchy distribution around a point

    :param samples: number of samples to generate
    :param mean: point to generate around
    :param variance: array of variances in each dimension
    :return: array of samples
    """
    n = len(mean)
    if variance is None:
        variance = np.ones(n)
    return np.transpose([sp_stats.cauchy.rvs(loc=mean[i], scale=variance[i], size=samples) for i in range(n)])


def truncation_selection(X, Y, elite):
    """
    Selects the best (lowest valued) samples from a list of samples

    :param X: array of samples
    :param Y: array of values
    :param elite: number of samples to select
    :return: array of best samples
    """
    indices = np.argsort(Y)
    selected = []
    for i in range(min(elite, len(indices))):
        selected.append(X[indices[i]])
    return selected


def tournament_selection(X, Y, elite, group_size=0):
    """
    Selects the best samples from successive randomly chosen groups

    :param X: array of samples
    :param Y: array of values
    :param elite: number of samples to select
    :param group_size: number of prospective samples in each group
    :return: array of best samples
    """
    if group_size == 0:
        group_size = elite
    selected = []
    for i in range(elite):
        indices = np.random.permutation(len(Y))[:group_size]
        group_x, group_y = X[indices], Y[indices]
        selected.append(group_x[np.argmin(group_y)])
    return np.array(selected)


def roulette_selection(X, Y, elite, fitness_function=None):
    """
    Selects samples with higher probability of selecting more fit samples

    :param X: array of samples
    :param Y: array of values
    :param elite: number of samples to select
    :param fitness_function: function that takes an array of values and returns an array of fitnesses
    :return: array of best samples
    """
    if fitness_function is None:
        fitness = np.max(Y) - np.array(Y)
    else:
        fitness = fitness_function(Y)
    selected = []
    for i in np.random.choice(range(len(Y)), size=elite, p=fitness / np.sum(fitness)):
        selected.append(X[i])
    return np.array(selected)


def single_point_crossover(x_0, x_1):
    """
    Takes the first m dimensions of x_0 and the last n - m dimensions of x_1. m is randomly generated

    :param x_0: first sample
    :param x_1: second sample
    :return: new sample
    """
    point = np.random.randint(0, len(x_0) + 1)
    return np.concatenate((x_0[:point], x_1[point:]))


def double_point_crossover(x_0, x_1):
    """
    Takes the first m dimensions of x_0, the next k - m dimensions of x_1, and the last n - k dimesions of x_0. m and k are randomly generated

    :param x_0: first sample
    :param x_1: second sample
    :return: new sample
    """
    points = np.random.randint(0, len(x_0) + 1, size=2)
    print(points)
    return np.concatenate((x_0[:min(points)], x_1[min(points):max(points)], x_0[max(points):]))


def uniform_crossover(x_0, x_1):
    """
    Has an equal chance of receiving each dimension from x_0 or x_1

    :param x_0: first sample
    :param x_1: second sample
    :return: new sample
    """
    n = len(x_0)
    child = np.zeros(n)
    for i in range(n):
        if np.random.rand() < 0.5:
            child[i] = x_0[i]
        else:
            child[i] = x_1[i]
    return child


def interpolation_crossover(x_0, x_1, scalar=0.5):
    """
    For real numbers, takes a point on the line between x_0 and x_1

    :param x_0: first sample
    :param x_1: second sample
    :param scalar: value in [0,1] to be closer to x_0 or x_1
    :return: new sample
    """
    return (1 - scalar) * np.array(x_0) + scalar * np.array(x_1)


def bitwise_mutation(x, probability=None):
    """
    Has a chance to flip the sign of any dimension

    :param x: sample
    :param probability: probability that each dimension gets mutated, defaults to 1 / n
    :return: mutated sample
    """
    n = len(x)
    y = x.copy()
    if probability is None:
        probability = 1 / n
    for i in range(n):
        if np.random.rand() < probability:
            y[i] *= -1
    return y


def gaussian_mutation(x, covariance=None):
    """
    Adds zero-mean gaussian noise to a sample

    :param x: sample
    :param covariance: covariance matrix for the noise
    :return: mutated sample
    """
    n = len(x)
    if covariance is None:
        covariance = np.identity(n)
    return x + np.random.multivariate_normal(np.zeros(n), covariance)


def genetic_selection(X, Y, elite, method, selection_args):
    """
    Wrapper for different selection methods

    :param X: array of samples
    :param Y: array of values
    :param elite: number of samples to select
    :param method: selection method to perform
    :param selection_args: dictionary of arguments for the selection method
    :return: array of selected elements
    """
    if method == "tournament":
        return tournament_selection(X, Y, elite, **selection_args)
    elif method == "roulette":
        return roulette_selection(X, Y, elite, **selection_args)
    else:
        return truncation_selection(X, Y, elite)


def genetic_crossover(x_0, x_1, method, crossover_args):
    """
    Wrapper for different crossover methods

    :param x_0: first sample
    :param x_1: second sample
    :param method: crossover method to perform
    :param crossover_args: dictionary of arguments for the crossover method
    :return: new sample
    """
    if method == "single_point":
        return single_point_crossover(x_0, x_1)
    elif method == "double_point":
        return double_point_crossover(x_0, x_1)
    elif method == "interpolation":
        return interpolation_crossover(x_0, x_1, **crossover_args)
    else:
        return uniform_crossover(x_0, x_1)


def genetic_mutation(x, method, mutation_args):
    """
    Wrapper for different mutation methods

    :param x: sample to mutate
    :param method: mutation method to perform
    :param mutation_args: dictionary of arguments for the mutation method
    :return: mutated sample
    """
    if method == "bitwise":
        return bitwise_mutation(x, **mutation_args)
    else:
        return gaussian_mutation(x, **mutation_args)


def genetic_method(f, f_args, x_0=None, init_method="cauchy", variance=None,
                   lower=None, upper=None, parent_count=10, max_steps=50,
                   selection="truncation", selection_args={},
                   crossover="uniform", crossover_args={},
                   mutation="gaussian", mutation_args={}, multithreaded=False):
    """
    Uses genetic evolution techniques to evolve a population of samples towards a local minimum

    :param f: objective function
    :param f_args: dictionary of immutable arguments for the objective function
    :param x_0: starting point for normal or cauchy distribution population initialization
    :param init_method: method of population initialization around a point
    :param variance: covariance matrix of array of variances in each dimension based on initialization method
    :param lower: lower bounds used in uniform population initialization
    :param upper: upper bounds used in uniform population initialization
    :param parent_count: number of samples to use at each step, explores m^2 children
    :param max_steps: number of generations to use
    :param selection: selection method to be used
    :param selection_args: dictionary of arguments for selection method
    :param crossover: crossover method to be used
    :param crossover_args: dictionary of arguments for crossover method
    :param mutation: mutation method to be used
    :param mutation_args: dictionary of arguments for mutation method
    :param multithreaded: whether the children of each generation are explored in parallel
    :return: a likely locally minimal design point
    """
    if x_0 is not None:
        if init_method == "normal":
            parents = random_normal(parent_count, x_0, variance)
        else:
            parents = random_cauchy(parent_count, x_0, variance)
    else:
        parents = random_uniform(parent_count, lower, upper)

    Y = batch_eval(f, parents, f_args, multithreaded)
    best_x = parents[np.argmin(Y)].copy()
    best_y = np.min(Y)

    while max_steps > 0:
        children = []
        for i in range(len(parents)):
            for j in range(len(parents)):
                children.append(genetic_crossover(parents[i], parents[j], crossover, crossover_args))
        for i in range(len(children)):
            children[i] = genetic_mutation(children[i], mutation, mutation_args)

        Y = batch_eval(f, children, f_args, multithreaded)
        if min(Y) < best_y:
            best_x, best_y = children[np.argmin(Y)].copy(), min(Y)

        parents = genetic_selection(children, Y, parent_count, selection, selection_args)
        max_steps -= 1
    return best_x


def differential_evolution(f, f_args, x_0=None, init_method="cauchy", variance=None,
                           lower=None, upper=None, population_count=50, crossover_prob=0.2, weight=0.4,
                           max_steps=50, multithreaded=False):
    """
    Uses interpolations of other samples in the population to slowly mutate points towards a local minimum

    :param f: objective function
    :param f_args: dictionary of immutable arguments for the objective function
    :param x_0: starting point for normal or cauchy distribution population initialization
    :param init_method: method of population initialization around a point
    :param variance: covariance matrix of array of variances in each dimension based on initialization method
    :param lower: lower bounds used in uniform population initialization
    :param upper: upper bounds used in uniform population initialization
    :param population_count: number of sample to maintain at each step
    :param crossover_prob: chance to use the interpolated sample's dimension
    :param weight: weight used in generating interpolated samples; s = a + weight * (b - c)
    :param max_steps: number of times to cycle and mutated each point
    :param multithreaded: whether the initial population will be explored in parralel
    :return: a likely locally minimal design point
    """
    if x_0 is not None:
        n = len(x_0)
        if init_method == "normal":
            samples = random_normal(population_count, x_0, variance)
        else:
            samples = random_cauchy(population_count, x_0, variance)
    else:
        n = len(lower)
        samples = random_uniform(population_count, lower, upper)
    values = batch_eval(f, samples, f_args, multithreaded)
    while max_steps > 0:
        for i in range(population_count):
            probs = np.ones(population_count)
            probs[i] = 0
            probs *= 1 / (population_count - 1)
            [a, b, c] = np.random.choice(population_count, size=3, replace=False, p=probs)
            z = samples[a] + weight * (samples[b] - samples[c])
            dim = np.random.randint(0, n)
            prospective = np.zeros(n)
            for j in range(n):
                if dim == j or np.random.rand() < crossover_prob:
                    prospective[j] = z[j]
                else:
                    prospective[j] = samples[i][j]
            new_value = f(prospective, **f_args)
            if new_value < values[i]:
                values[i] = new_value
                samples[i] = prospective.copy()
        max_steps -= 1
    return samples[np.argmin(values)]


def particle_swarm(f, f_args, x_0=None, init_method="cauchy", variance=None,
                   lower=None, upper=None, population_count=50,
                   velocity_weight=1, personal_weight=1, universal_weight=1, max_steps=50, multithreaded=False):
    """
    Updates the velocity of each sample to draw it towards both the universal best sample point and it's personal best sample point

    :param f: objective function
    :param f_args: dictionary of immutable arguments for the objective function
    :param x_0: starting point for normal or cauchy distribution population initialization
    :param init_method: method of population initialization around a point
    :param variance: covariance matrix of array of variances in each dimension based on initialization method
    :param lower: lower bounds used in uniform population initialization
    :param upper: upper bounds used in uniform population initialization
    :param population_count: number of sample to maintain at each step
    :param velocity_weight: scalar to be used when updating the sample with it's velocity
    :param personal_weight: scalar to be used when updating velocities toward the personal best sample point
    :param universal_weight: scalar to be used when updating velocities toward the universal best sample point
    :param max_steps: number of times to update each sample
    :param multithreaded: whether the initial population will be explored in parallel
    :return: a likely locally minimal design point
    """
    if x_0 is not None:
        n = len(x_0)
        if init_method == "normal":
            samples = random_normal(population_count, x_0, variance)
        else:
            samples = random_cauchy(population_count, x_0, variance)
    else:
        n = len(lower)
        samples = random_uniform(population_count, lower, upper)
    velocities = np.zeros((population_count, n))
    personal_best = samples.copy()
    personal_values = batch_eval(f, samples, f_args, multithreaded)
    universal_best = samples[np.argmin(personal_values)].copy()
    universal_value = np.min(personal_values)
    while max_steps > 0:
        for i in range(population_count):
            samples[i] = samples[i] + velocities[i]
            value = f(samples[i], **f_args)
            if value < personal_values[i]:
                personal_best[i] = samples[i].copy()
                personal_values[i] = value
            if value < universal_value:
                universal_best = samples[i].copy()
                universal_value = value
            r0, r1 = np.random.rand(), np.random.rand()
            velocities[i] = velocity_weight * velocities[i] + personal_weight * r0 * (personal_best[i] - samples[i]) + \
                            universal_weight * r1 * (universal_best - samples[i])
        max_steps -= 1
    return universal_best


def firefly_method(f, f_args, x_0=None, init_method="cauchy", variance=None,
                   lower=None, upper=None, population_count=50, intensity=1, absorption=0.5, step_size=0.1,
                   max_steps=50, multithreaded=False):
    """
    Each less fit sample gets drawn to each sample more fit than it according to the distance between them

    :param f: objective function
    :param f_args: dictionary of immutable arguments for the objective function
    :param x_0: starting point for normal or cauchy distribution population initialization
    :param init_method: method of population initialization around a point
    :param variance: covariance matrix of array of variances in each dimension based on initialization method
    :param lower: lower bounds used in uniform population initialization
    :param upper: upper bounds used in uniform population initialization
    :param population_count: number of sample to maintain at each step
    :param intensity: scalar to detemine weight of each update
    :param absorption: larger values decrease the attraction for more distant samples
    :param step_size: amount of random gaussian noise to move with each update
    :param max_steps: number of times each sample gets drawn towards each other sample, roughly n^2 updates
    :param multithreaded: whether the initial population will be explored in parallel
    :return: a likely locally minimal design point
    """
    if x_0 is not None:
        n = len(x_0)
        if init_method == "normal":
            samples = random_normal(population_count, x_0, variance)
        else:
            samples = random_cauchy(population_count, x_0, variance)
    else:
        n = len(lower)
        samples = random_uniform(population_count, lower, upper)
    values = batch_eval(f, samples, f_args, multithreaded)
    while max_steps > 0:
        for i in range(population_count):
            for j in range(i+1, population_count):
                dist = np.linalg.norm(samples[i] - samples[j])
                if values[i] < values[j]:
                    # Move j towards i and re-evaluate that value
                    samples[j] = samples[j] + intensity * np.exp(-absorption * dist) * dist * (samples[i] - samples[j]) + \
                        step_size * np.random.multivariate_normal(np.zeros(n), np.identity(n))
                    values[j] = f(samples[j], **f_args)
                else:
                    # Move i towards j and re-evaluate that value
                    samples[i] = samples[i] + intensity * np.exp(-absorption * dist) * dist * (samples[j] - samples[i]) + \
                                 step_size * np.random.multivariate_normal(np.zeros(n), np.identity(n))
                    values[i] = f(samples[i], **f_args)
        max_steps -= 1
    return samples[np.argmin(values)]


def cuckoo_search(f, f_args, x_0=None, init_method="cauchy", variance=None,
                   lower=None, upper=None, population_count=50, max_steps=50, multithreaded=False):
    """
    Each bird lays 2 eggs at a random point nearby and half of the eggs die according to a roulette selection

    :param f: objective function
    :param f_args: dictionary of immutable arguments for the objective function
    :param x_0: starting point for normal or cauchy distribution population initialization
    :param init_method: method of population initialization around a point
    :param variance: covariance matrix of array of variances in each dimension based on initialization method
    :param lower: lower bounds used in uniform population initialization
    :param upper: upper bounds used in uniform population initialization
    :param population_count: number of sample to maintain at each step
    :param max_steps: number of generations to explore
    :param multithreaded: whether each generation gets explored in parallel
    :return: a likely locally minimal design point
    """
    if x_0 is not None:
        if init_method == "normal":
            eggs = random_normal(population_count * 2, x_0, variance)
        else:
            eggs = random_cauchy(population_count * 2, x_0, variance)
    else:
        eggs = random_uniform(population_count * 2, lower, upper)
    values = batch_eval(f, eggs, f_args, multithreaded)
    best_x = eggs[np.argmin(values)]
    best_y = np.min(values)

    birds = roulette_selection(eggs, values, population_count)
    while max_steps > 0:
        eggs = []
        for i in range(population_count):
            babies = random_cauchy(2, birds[i], variance)
            for egg in babies:
                eggs.append(egg.copy())
        values = batch_eval(f, eggs, f_args, multithreaded)
        if np.min(values) < best_y:
            best_x = eggs[np.argmin(values)].copy()
            best_y = np.min(values)
        birds = roulette_selection(eggs, values, population_count)
        max_steps -= 1
    return best_x
