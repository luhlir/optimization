import numpy as np
import itertools


def stratify_samples(indices):
    """
    Randomly distributes a set of points within their designated subspace

    :param indices: indices of the samples designating their subspace
    :return: the stratified samples' indices
    """
    indices = np.array(indices, dtype='float64')
    for index in indices:
        index += np.random.rand(len(index)) - 0.5
    return indices


def volume(samples):
    """
    Calculates the volume of the hyper-rectangle created by a group of points

    :param samples: the sample points
    :return: the volume of the hyper-rectangle
    """
    minimum = np.min(samples, axis=0)
    maximum = np.max(samples, axis=0)
    prod = np.cumprod(minimum - maximum)
    return np.abs(prod[-1]), minimum, maximum


def translate_indices(input_points, left_out):
    """
    Shift indices accounting for indices left out of calculation

    :param input_points: input indices not accounting for left out indices
    :param left_out: indices not included while generating input indices
    :return: the shifted indices
    """
    sort_left = np.sort(left_out)
    output = np.sort(input_points)
    ones = np.ones(len(output), dtype='int64')
    i = 0
    j = 0
    while j < len(sort_left):
        if output[i] >= sort_left[j]:
            output += ones
            j += 1
        else:
            ones[i] = 0
            i += 1
        if i == len(output):
            break
    return output


recursion_count = 0


def discrepancy_recursion(samples, left_out, previously_explored, sample_count, total_points, curr_max, lower, upper):
    """
    Recursively finds the discrepancy of a subset of points

    :param samples: the original array of points
    :param left_out: indices of points to be left out of this calculation
    :param previously_explored: set of groups of indices that have already been left out
    :param sample_count: number of true samples (ignoring shadow border points)
    :param total_points: total number of points at the start
    :param curr_max: the currently known maximum for the whole system
    :param lower: index of lower bound in samples array
    :param upper: index of upper bound in samples array
    :return: the discrepancy of the subset of points
    """
    global recursion_count
    if len(samples) - len(left_out) < 2:
        return -1, [], []
    working_samples = np.delete(samples, list(left_out), axis=0)
    current_volume, max_lower, max_upper = volume(working_samples)
    if current_volume < curr_max and sample_count / total_points < curr_max:
        return -1, [], []
    max_disc = np.abs(sample_count / total_points - current_volume)
    # Determine the number of points on the border of this and count down from there
    border_points_max = np.argmax(working_samples, axis=0)
    border_points_min = np.argmin(working_samples, axis=0)
    border_points = set()
    for point in border_points_min:
        border_points.add(point)
    for point in border_points_max:
        border_points.add(point)
    corrected_points = translate_indices(list(border_points), list(left_out))
    # Try to remove all of the border points and see if that's even better
    boundary = 0
    if lower in corrected_points:
        boundary += 1
    if upper in corrected_points:
        boundary += 1
    new_disc = np.abs((sample_count - min(len(border_points) - boundary, sample_count)) / total_points - current_volume)
    if new_disc > max_disc:
        max_disc = new_disc
    # Remove each point on the border and call recursively down
    for point in corrected_points:
        left_out.add(point)
        for i in range(len(previously_explored)):
            if len(left_out ^ previously_explored[i]) == 0:
                break
        else:
            previously_explored.append(left_out.copy())
            if point == lower or point == upper:
                new_disc, temp_lower, temp_upper = discrepancy_recursion(samples, left_out, previously_explored,
                                                                 sample_count, total_points,
                                                                 max(max_disc, curr_max), lower, upper)
            else:
                new_disc, temp_lower, temp_upper = discrepancy_recursion(samples, left_out, previously_explored,
                                                                 sample_count - 1, total_points,
                                                                 max(max_disc, curr_max), lower, upper)
            if new_disc > max_disc:
                max_disc, max_lower, max_upper = new_disc, temp_lower.copy(), temp_upper.copy()
        left_out.discard(point)
    return max_disc, max_lower, max_upper


def samples_discrepancy(samples, lower_bound=None, upper_bound=None):
    """
    Calculates the discrepancy of a set of points

    :param samples: array of sample points
    :param lower_bound: lower bound of search space, default to minimum of sample points
    :param upper_bound: upper bound of search space, default to maximum of sample points
    :return: the discrepancy of the sample points
    """
    global recursion_count
    sample_count = len(samples)

    if lower_bound is None:
        lower_bound = np.min(samples, axis=0)
        lower = -1
    else:
        lower = len(samples)
        samples = np.append(samples, [lower_bound], axis=0)
    if upper_bound is None:
        upper_bound = np.max(samples, axis=0)
        upper = -1
    else:
        upper = len(samples)
        samples = np.append(samples, [upper_bound], axis=0)
    # Project all points to the unit hyper-rectangle
    corrected_samples = (np.array(samples) - lower_bound) / (np.array(upper_bound) - lower_bound)
    max_disc, max_lower, max_upper = discrepancy_recursion(corrected_samples, set(), [], sample_count, sample_count, 0, lower, upper)

    return max_disc, max_lower * (np.array(upper_bound) - lower_bound) + lower_bound, \
           max_upper * (np.array(upper_bound) - lower_bound) + lower_bound


def full_factorial_samples(lower_bound, upper_bound, samp_per_dimension, stratified=False):
    """
    Generates a full factorial sample plan in a given search space

    :param lower_bound: lower bound of the search space
    :param upper_bound: upper bound of the search space
    :param samp_per_dimension: number of samples to generate in each dimension, n^d total samples
    :param stratified: whether the sample points will be uniformly stratified in their subspaces
    :return: a set of sample points
    """
    if not isinstance(samp_per_dimension, list) and not isinstance(samp_per_dimension, np.ndarray):
        samp_per_dimension = [samp_per_dimension] * len(lower_bound)
    ind_lists = []
    for m in samp_per_dimension:
        ind_lists.append(range(m))
    ind_prod = itertools.product(*ind_lists)
    indices = []
    for index in ind_prod:
        indices.append(list(index))
    if stratified:
        indices = stratify_samples(indices)
    samples = []
    for index in indices:
        samples.append(lower_bound + (index + np.ones(len(index)) * 0.5) * (np.array(upper_bound) - lower_bound) / samp_per_dimension)
    return np.array(samples)


def uniform_projection_samples(lower_bound, upper_bound, samp_per_dimension=None, stratified=False):
    """
    Generates a uniform projection sample plan in a given search space

    :param lower_bound: lower bound of the search space
    :param upper_bound: upper bound of the search space
    :param samp_per_dimension: number of samples to generate
    :param stratified: whether the sample points will be uniformly stratified in their subspaces
    :return: a set of sample points
    """
    if samp_per_dimension is None:
        samp_per_dimension = len(lower_bound)
    elif samp_per_dimension < len(lower_bound):
        return []
    indices = []
    for n in range(len(lower_bound)):
        indices.append(np.random.permutation(samp_per_dimension))
    indices = np.array(indices).transpose()
    if stratified:
        indices = stratify_samples(indices)
    samples = []
    for index in indices:
        samples.append(lower_bound + (index + np.ones(len(index)) * 0.5) * (np.array(upper_bound) - lower_bound) / samp_per_dimension)
    return np.array(samples)


def samples_pairwise_distance(samples, order=2, matrix=False):
    """
    Calculates the pairwise distances for a set of sample points

    :param samples: array of sample points
    :param order: the order of p-norm to use in the calculation
    :param matrix: whether the return value should be a symmetrical nxn matrix
    :return: a list of pairwise distances
    """
    if not matrix:
        distances = []
        for i in range(len(samples)):
            for j in range(i + 1, len(samples)):
                distances.append(np.linalg.norm(samples[i] - samples[j], ord=order))
    else:
        distances = np.zeros((len(samples), len(samples)))
        for i in range(len(samples)):
            for j in range(i + 1, len(samples)):
                dist = np.linalg.norm(samples[i] - samples[j], ord=order)
                distances[i][j] = dist
                distances[j][i] = dist
    return distances


def compare_pairwise_distances(samples_1, samples_2, order):
    """
    Compares two sets of sample points based on pairwise distances

    :param samples_1: first set of samples
    :param samples_2: second set of samples
    :param order: order of p-norm to use in the distance calculation
    :return: 1 if samples_1 is a better set, -1 if samples_2 is a better set, and 0 if they are equal
    """
    dist_1 = np.sort(samples_pairwise_distance(samples_1, order))
    dist_2 = np.sort(samples_pairwise_distance(samples_2, order))
    for i in range(min(len(dist_1), len(dist_2))):
        if dist_1[i] < dist_2[i]:
            return 1
        elif dist_1[i] > dist_2[i]:
            return -1
    if len(samples_1) > len(samples_2):
        return 1
    elif len(samples_1) < len(samples_2):
        return -1
    else:
        return 0


def samples_morris_mitchell(samples, exp=None, order=2):
    """
    Calculates the Morris-Mitchell criterion for a set of sample points

    :param samples: the set of sample points
    :param exp: exponent to use in calculation, defaults to maximizing over a range of exponents
    :param order: order to use in calculating the pairwise distances
    :return: the Morris-Mitchell criterion
    """
    distances = samples_pairwise_distance(samples, order)
    if exp is None:
        exp = [1, 2, 5, 10, 20, 50, 100]
        maximum = 0
        for q in exp:
            phi = sum([d ** -q for d in distances]) ** (1 / q)
            if phi > maximum:
                maximum = phi
    else:
        maximum = sum([d ** -exp for d in distances]) ** (1 / exp)
    return maximum


def brute_force_space_filling(samples, subset_size, order=2):
    """
    Finds the most space-filling subset of sample points

    :param samples: whole set of sample points
    :param subset_size: number of sample points to include in the subset
    :param order: order of norm to use in the distance calculations
    :return: array of best space-filling points
    """
    # TODO: Try every single n choose p of the set. How to generate all groups?
    return np.inf


def greedy_space_filling(samples, subset_size, order=2):
    choice = np.random.randint(0, len(samples))
    choice_array = [choice]
    distances = samples_pairwise_distance(samples, order, True)
    subset = np.array([distances[choice]])
    while len(subset) < subset_size:
        distances[:, choice] = np.nan
        distances[choice, :] = np.nan
        distances[choice, choice] = np.inf
        sub_min = np.nanmin(subset, axis=0)
        mini = np.inf
        choice = -1
        for i in range(len(distances)):
            maxi = -np.inf
            for j in range(len(distances[i])):
                checker = np.nanmin([distances[i][j], sub_min[j]])
                if checker > maxi:
                    maxi = checker
            if maxi < mini:
                mini = maxi
                choice = i
        subset = np.append(subset, [distances[choice]], axis=0)
        choice_array.append(choice)
    return samples[choice_array]


# TODO: Implement Exchange Algorithm (pg 246)


# TODO: Implement Additive Recurrence Method (pg 248)


# TODO: Implement Halton and Sobol Sequence Methods (pg 249)
