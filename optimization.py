from enum import Enum

import firstOrderMethods as fom
import secondOrderMethods as som
import zeroOrderMethods as zom
import stochasticMethods as stm
import populationMethods as pop
import constraintMethods as con
import multiobjectiveMethods as mul
import gaussianProcess as gp


class Method(Enum):
    """
    Supported Methods for General Optimization
    """
    GRADIENT_DESCENT = fom.gradient_descent
    """
    Uses a simple gradient of the objective function to decide which direction to step in. Takes the following arguments:
    \nf_prime - gradient of the objective function
    \nfp_args - dictionary of immutable arguments for the gradient function
    \nauto_diff - if True, will not use gradient function and will use automatic differentiation instead
    \nmax_steps - maximum number of steps to take before returning
    \ntol - necessary distance to travel at each step before returning
    \nlin_method - line search method used to determine step size
    \nlin_args - dictionary of immutable arguments for the line search function
    """
    CONJUGATE_DESCENT = fom.conjugate_descent
    """
    Uses previous gradient information to determine which direction to search. Takes the following arguments:
    \nf_prime - gradient of the objective function
    \nfp_args - dictionary of immutable arguments for the gradient function
    \nauto_diff - if True, will not use gradient function and will use automatic differentiation instead
    \nbeta_method - "polak-ribiere" or "fletcher-reeves" for adding previous gradient information to direction
    \nmax_steps - maximum number of steps to take before returning
    \ntol - necessary distance to travel at each step before returning
    \nlin_method - line search method used to determine step size
    \nlin_args - dictionary of immutable arguments for the line search function
    """
    MOMENTUM_DESCENT = fom.momentum_descent
    """
    Updates a momentum vector with gradient information to determine direction and size of each step. Takes the following arguments:
    \nf_prime - gradient of the objective function
    \nfp_args - dictionary of immutable arguments for the gradient function
    \nauto_diff - if True, will not use gradient function and will use automatic differentiation instead
    \nalpha - scalar for current gradient
    \nbeta - scalar for previous momentum
    \nmax_steps - maximum number of steps to take before returning
    \ntol - necessary distance to travel at each step before returning
    """
    NESTEROV_DESCENT = fom.nesterov_descent
    """
    Updates a momentum vector with future gradient information to determine direction and size of each step. Takes the following arguments:
    \nf_prime - gradient of the objective function
    \nfp_args - dictionary of immutable arguments for the gradient function
    \nauto_diff - if True, will not use gradient function and will use automatic differentiation instead
    \nalpha - scalar for future gradient
    \nbeta - scalar for previous momentum
    \nmax_steps - maximum number of steps to take before returning
    \ntol - necessary distance to travel at each step before returning
    """
    ADAGRAD_DESCENT = fom.adagrad_descent
    """
    Updates a momentum vector with squared gradient to determine direction and size of each step. Takes the following arguments:
    \nf_prime - gradient of the objective function
    \nfp_args - dictionary of immutable arguments for the gradient function
    \nauto_diff - if True, will not use gradient function and will use automatic differentiation instead
    \nalpha - scalar for momentum and gradient toward step size
    \nmax_steps - maximum number of steps to take before returning
    \ntol - necessary distance to travel at each step before returning
    """
    RMSPROP_DESCENT = fom.rmsprop_descent
    """
    Updates a momentum vector using a weighted sum of previous momentum and gradient information to determine direction and size of each step
    \nf_prime - gradient of the objective function
    \nfp_args - dictionary of immutable arguments for the gradient function
    \nauto_diff - if True, will not use gradient function and will use automatic differentiation instead
    \nalpha - scalar for momentum towards step size
    \ngamma - scalar for weight of momentum in momentum update
    \nmax_steps - maximum number of steps to take before returning
    \ntol - necessary distance to travel at each step before returning
    """
    ADADELTA_DESCENT = fom.adadelta_descent
    """
    Updates a momenum vector with two weighted sums of gradients and momentums to determine direction and size of each step. Takes the following arguments:
    \nf_prime - gradient of the objective function
    \nfp_args - dictionary of immutable arguments for the gradient function
    \nauto_diff - if True, will not use gradient function and will use automatic differentiation instead
    \ndelta - weight of previous step direction/size in step direction/size update
    \ngamma - weight of previous momentum in momentum update
    \nmax_steps - maximum number of steps to take before returning
    \ntol - necessary distance to travel at each step before returning
    """
    ADAM_DESCENT = fom.adam_descent
    """
    Updates two momentum vectors that increase over time to determine direction and size of each step. Take the following arguments:
    \nf_prime - gradient of the objective function
    \nfp_args - dictionary of immutable arguments for the gradient function
    \nauto_diff - if True, will not use gradient function and will use automatic differentiation instead
    \nalpha - scalar for momentum in step direction/size
    \ndelta - weight for current momentum in first momentum vector update
    \ngamma - weight for current momentum in second momentum vector update
    \nmax_steps - maximum number of steps to take before returning
    \ntol - necessary distance to travel at each step before returning
    """
    HYPERGRADIENT_DESCENT = fom.hypergradient_descent
    """
    Uses previous gradient information to update scalar "alpha" to determine step size; steps in the direction of the gradient. Takes the following arguments:
    \nf_prime - gradient of the objective function
    \nfp_args - dictionary of immutable arguments for the gradient function
    \nauto_diff - if True, will not use gradient function and will use automatic differentiation instead
    \nalpha - initial step size value
    \nmu - scalar used in step size update
    \nmax_steps - maximum number of steps to take before returning
    \ntol - necessary distance to travel at each step before returning
    """
    NEWTONS_METHOD = som.newtons_method
    """
    Uses the gradient and inverse of the Hessian to determine a direction to search for a minimum on. Takes the following arguments:
    \nf_prime - gradient function of the objective function
    \nfp_args - dictionary of immutable arguments for the gradient function
    \nf_dprime - Hessian function of the objective function
    \nfdp_args - dictionary of immutable arguments for the Hessian function
    \nauto_diff - if True, will not use gradient or Hessian functions and will use automatic differentiation instead
    \nmax_steps - maximum number of steps to take before returning
    \ntol - necessary distance to travel at each step before returning
    \nlin_method - line search method to use
    \nlin_args - dictionary of immutable arguments for the line search method
    """
    QUASI_NEWTONS_METHOD = som.quasi_newtons_method
    """
    Uses the gradient and an estimated inverse of the Hessian to determine a direction to search for a minimum in. Takes the following arguments:
    \nf_prime - gradient function of the objective function
    \nfp_args - dictionary of immutable arguments for the gradient function
    \nauto_diff - if True, will not use gradient function and will use automatic differentiation instead
    \nappr - inverse Hessian approximation method "dfp" or "???"
    \nmax_steps - maximum number of steps to take before returning
    \ntol - necessary distance to travel at each step before returning
    \nlin_method - line search method to use
    \nlin_args - dictionary of immutable arguments for the line search method
    """
    COORDINATE_DESCENT = zom.coordinate_descent
    """
    Conjugate Descent takes the following arguments:
    """
    POWELLS_METHOD = zom.powells_method
    """
    Conjugate Descent takes the following arguments:
    """
    HOOKE_JEEVES_METHOD = zom.hooke_jeeves_method
    """
    Conjugate Descent takes the following arguments:
    """
    PATTERN_SEARCH = zom.pattern_search
    """
    Conjugate Descent takes the following arguments:
    """
    NELDER_MEAD_SIMPLEX = zom.nelder_mead_simplex
    """
    Conjugate Descent takes the following arguments:
    """
    DIVIDED_RECTANGLES = zom.divided_rectangles
    """
    Conjugate Descent takes the following arguments:
    """
    NOISY_DESCENT = stm.noisy_descent
    """
    Conjugate Descent takes the following arguments:
    """
    MESH_ADAPTIVE_SEARCH = stm.mesh_adaptive_search
    """
    Conjugate Descent takes the following arguments:
    """
    SIMULATED_ANNEALING = stm.simulated_annealing
    """
    Conjugate Descent takes the following arguments:
    """
    CORANA_ANNEALING = stm.corana_annealing
    """
    Conjugate Descent takes the following arguments:
    """
    CROSS_ENTROPY = stm.cross_entropy
    """
    Conjugate Descent takes the following arguments:
    """
    COVARIANCE_MATRIX_ADAPTATION = stm.covariance_matrix_adaptation
    """
    Conjugate Descent takes the following arguments:
    """
    GENETIC_METHOD = pop.genetic_method
    """
    Conjugate Descent takes the following arguments:
    """
    DIFFERENTIAL_EVOLUTION = pop.differential_evolution
    """
    Conjugate Descent takes the following arguments:
    """
    PARTICLE_SWARM = pop.particle_swarm
    """
    Conjugate Descent takes the following arguments:
    """
    FIREFLY_METHOD = pop.firefly_method
    """
    Conjugate Descent takes the following arguments:
    """
    CUCKOO_SEARCH = pop.cuckoo_search
    """
    Conjugate Descent takes the following arguments:
    """
    HYPERRECTANGLE_CONSTRAINT = con.hyperrectangle_constraint
    """
    Conjugate Descent takes the following arguments:
    """
    PENALTY_CONSTRAINT = con.penalty_constraint
    """
    Conjugate Descent takes the following arguments:
    """
    AUGMENTED_LAGRANGE = con.augmented_lagrange
    """
    Conjugate Descent takes the following arguments:
    """
    INTERIOR_POINT_METHOD = con.interior_point_method
    """
    Conjugate Descent takes the following arguments:
    """
    NAIVE_PARETO = mul.naive_pareto
    """
    Conjugate Descent takes the following arguments:
    """
    WEIGHTED_PARETO_METHOD = mul.weighted_pareto_method
    """
    Conjugate Descent takes the following arguments:
    """
    WEIGHTED_PARETO_SCAN_METHOD = mul.weighted_pareto_scan_method
    """
    Conjugate Descent takes the following arguments:
    """
    GOAL_METHOD = mul.goal_method
    """
    Conjugate Descent takes the following arguments:
    """
    WEIGHTED_GOAL_METHOD = mul.weighted_goal_method
    """
    Conjugate Descent takes the following arguments:
    """
    WEIGHTED_GOAL_SCAN_METHOD = mul.weighted_goal_scan_method
    """
    Conjugate Descent takes the following arguments:
    """
    WEIGHTED_MIN_MAX_METHOD = mul.weighted_min_max_method
    """
    Conjugate Descent takes the following arguments:
    """
    WEIGHTED_MIN_MAX_SCAN_METHOD = mul.weighted_min_max_scan_method
    """
    Conjugate Descent takes the following arguments:
    """
    EXPONENTIAL_WEIGHT_METHOD = mul.exponential_weight_method
    """
    Conjugate Descent takes the following arguments:
    """
    EXPONENTIAL_WEIGHT_SCAN_METHOD = mul.exponential_weight_scan_method
    """
    Conjugate Descent takes the following arguments:
    """
    VECTOR_EVALUATED_GENETIC_METHOD = pop.vector_evaluated_genetic_method
    """
    Conjugate Descent takes the following arguments:
    """
    AUTO_GAUSSIAN_PROCESS = gp.auto_gaussian_process
    """
    Conjugate Descent takes the following arguments:
    """


def optimize(f, x_0, f_args={}, opt_method=Method.GRADIENT_DESCENT, **opt_args):
    """
    Attempts to find a locally minimum design point of the design point using the specified optimization method

    :param f: design point
    :param x_0: initial design point if the optimization method takes one. None is acceptable if using upper/lower bounds
    :param f_args: dictionary of immutable arguments for the objective function
    :param opt_method: Method enum being used for the optimization
    :param opt_args: any extra arguments that the optimization method takes
    :return: a likely locally minimum design point
    """

    # Some methods don't take the same order of arguments, so they need to be handled specially
    if f_args is None:
        f_args = {}
    match opt_method:
        case Method.NELDER_MEAD_SIMPLEX:
            return zom.nelder_mead_simplex(f, f_args, x_0=x_0, **opt_args)
        case Method.DIVIDED_RECTANGLES:
            return zom.divided_rectangles(f, f_args, **opt_args)
        case Method.CROSS_ENTROPY:
            return stm.cross_entropy(f, f_args, x_0=x_0, **opt_args)
        case Method.GENETIC_METHOD:
            return pop.genetic_method(f, f_args, x_0=x_0, **opt_args)
        case Method.DIFFERENTIAL_EVOLUTION:
            return pop.differential_evolution(f, f_args, x_0=x_0, **opt_args)
        case Method.PARTICLE_SWARM:
            return pop.particle_swarm(f, f_args, x_0=x_0, **opt_args)
        case Method.FIREFLY_METHOD:
            return pop.firefly_method(f, f_args, x_0=x_0, **opt_args)
        case Method.CUCKOO_SEARCH:
            return pop.cuckoo_search(f, f_args, x_0=x_0, **opt_args)
        case Method.NAIVE_PARETO:
            return mul.naive_pareto(f, f_args, x_0=x_0, **opt_args)
        case Method.VECTOR_EVALUATED_GENETIC_METHOD:
            return pop.vector_evaluated_genetic_method(f, f_args, x_0=x_0, **opt_args)
        case Method.AUTO_GAUSSIAN_PROCESS:
            return gp.auto_gaussian_process(f, f_args, x_0, **opt_args)
        case _:
            return opt_method.__call__(f, x_0, f_args, **opt_args)
