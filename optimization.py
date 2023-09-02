from enum import Enum, auto

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
    GRADIENT_DESCENT = auto()
    """
    Uses a simple gradient of the objective function to decide which direction to step in. Takes the following arguments:
    f_prime - gradient of the objective function
    fp_args - dictionary of immutable arguments for the gradient function
    auto_diff - if True, will not use gradient function and will use automatic differentiation instead
    max_steps - maximum number of steps to take before returning
    tol - necessary distance to travel at each step before returning
    lin_method - line search method used to determine step size
    lin_args - dictionary of immutable arguments for the line search function
    """
    CONJUGATE_DESCENT = auto()
    """
    Uses previous gradient information to determine which direction to search. Takes the following arguments:
    f_prime - gradient of the objective function
    fp_args - dictionary of immutable arguments for the gradient function
    auto_diff - if True, will not use gradient function and will use automatic differentiation instead
    beta_method - "polak-ribiere" or "fletcher-reeves" for adding previous gradient information to direction
    max_steps - maximum number of steps to take before returning
    tol - necessary distance to travel at each step before returning
    lin_method - line search method used to determine step size
    lin_args - dictionary of immutable arguments for the line search function
    """
    MOMENTUM_DESCENT = auto()
    """
    Conjugate Descent takes the following arguments:
    """
    NESTEROV_DESCENT = auto()
    """
    Conjugate Descent takes the following arguments:
    """
    ADAGRAD_DESCENT = auto()
    """
    Conjugate Descent takes the following arguments:
    """
    RMSPROP_DESCENT = auto()
    """
    Conjugate Descent takes the following arguments:
    """
    ADADELTA_DESCENT = auto()
    """
    Conjugate Descent takes the following arguments:
    """
    ADAM_DESCENT = auto()
    """
    Conjugate Descent takes the following arguments:
    """
    HYPERGRADIENT_DESCENT = auto()
    """
    Conjugate Descent takes the following arguments:
    """
    NEWTONS_METHOD = auto()
    """
    Conjugate Descent takes the following arguments:
    """
    QUASI_NEWTONS_METHOD = auto()
    """
    Conjugate Descent takes the following arguments:
    """
    COORDINATE_DESCENT = auto()
    """
    Conjugate Descent takes the following arguments:
    """
    POWELLS_METHOD = auto()
    """
    Conjugate Descent takes the following arguments:
    """
    HOOKE_JEEVES_METHOD = auto()
    """
    Conjugate Descent takes the following arguments:
    """
    PATTERN_SEARCH = auto()
    """
    Conjugate Descent takes the following arguments:
    """
    NELDER_MEAD_SIMPLEX = auto()
    """
    Conjugate Descent takes the following arguments:
    """
    DIVIDED_RECTANGLES = auto()
    """
    Conjugate Descent takes the following arguments:
    """
    NOISY_DESCENT = auto()
    """
    Conjugate Descent takes the following arguments:
    """
    MESH_ADAPTIVE_SEARCH = auto()
    """
    Conjugate Descent takes the following arguments:
    """
    SIMULATED_ANNEALING = auto()
    """
    Conjugate Descent takes the following arguments:
    """
    CORANA_ANNEALING = auto()
    """
    Conjugate Descent takes the following arguments:
    """
    CROSS_ENTROPY = auto()
    """
    Conjugate Descent takes the following arguments:
    """
    COVARIANCE_MATRIX_ADAPTATION = auto()
    """
    Conjugate Descent takes the following arguments:
    """
    GENETIC_METHOD = auto()
    """
    Conjugate Descent takes the following arguments:
    """
    DIFFERENTIAL_EVOLUTION = auto()
    """
    Conjugate Descent takes the following arguments:
    """
    PARTICLE_SWARM = auto()
    """
    Conjugate Descent takes the following arguments:
    """
    FIREFLY_METHOD = auto()
    """
    Conjugate Descent takes the following arguments:
    """
    CUCKOO_SEARCH = auto()
    """
    Conjugate Descent takes the following arguments:
    """
    HYPERRECTANGLE_CONSTRAINT = auto()
    """
    Conjugate Descent takes the following arguments:
    """
    PENALTY_CONSTRAINT = auto()
    """
    Conjugate Descent takes the following arguments:
    """
    AUGMENTED_LAGRANGE = auto()
    """
    Conjugate Descent takes the following arguments:
    """
    INTERIOR_POINT_METHOD = auto()
    """
    Conjugate Descent takes the following arguments:
    """
    NAIVE_PARETO = auto()
    """
    Conjugate Descent takes the following arguments:
    """
    WEIGHTED_PARETO_METHOD = auto()
    """
    Conjugate Descent takes the following arguments:
    """
    WEIGHTED_PARETO_SCAN_METHOD = auto()
    """
    Conjugate Descent takes the following arguments:
    """
    GOAL_METHOD = auto()
    """
    Conjugate Descent takes the following arguments:
    """
    WEIGHTED_GOAL_METHOD = auto()
    """
    Conjugate Descent takes the following arguments:
    """
    WEIGHTED_GOAL_SCAN_METHOD = auto()
    """
    Conjugate Descent takes the following arguments:
    """
    WEIGHTED_MIN_MAX_METHOD = auto()
    """
    Conjugate Descent takes the following arguments:
    """
    WEIGHTED_MIN_MAX_SCAN_METHOD = auto()
    """
    Conjugate Descent takes the following arguments:
    """
    EXPONENTIAL_WEIGHT_METHOD = auto()
    """
    Conjugate Descent takes the following arguments:
    """
    EXPONENTIAL_WEIGHT_SCAN_METHOD = auto()
    """
    Conjugate Descent takes the following arguments:
    """
    VECTOR_EVALUATED_GENETIC_METHOD = auto()
    """
    Conjugate Descent takes the following arguments:
    """
    AUTO_GAUSSIAN_PROCESS = auto()
    """
    Conjugate Descent takes the following arguments:
    """


def optimize(f, x_0, f_args={}, opt_method=Method.GRADIENT_DESCENT, **opt_args):

    match opt_method:
        case Method.GRADIENT_DESCENT:
            return fom.gradient_descent(f, x_0, f_args, **opt_args)
        case Method.CONJUGATE_DESCENT:
            return fom.conjugate_descent(f, x_0, f_args, **opt_args)
        case Method.MOMENTUM_DESCENT:
            return fom.momentum_descent(f, x_0, f_args, **opt_args)
        case Method.NESTEROV_DESCENT:
            return fom.momentum_descent(f, x_0, f_args, **opt_args)
        case Method.ADAGRAD_DESCENT:
            return fom.momentum_descent(f, x_0, f_args, **opt_args)
        case Method.RMSPROP_DESCENT:
            return fom.momentum_descent(f, x_0, f_args, **opt_args)
        case Method.ADADELTA_DESCENT:
            return fom.momentum_descent(f, x_0, f_args, **opt_args)
        case Method.ADAM_DESCENT:
            return fom.momentum_descent(f, x_0, f_args, **opt_args)
        case Method.HYPERGRADIENT_DESCENT:
            return fom.momentum_descent(f, x_0, f_args, **opt_args)
        case Method.NEWTONS_METHOD:
            return som.newtons_method(f, x_0, f_args, **opt_args)
        case Method.QUASI_NEWTONS_METHOD:
            return som.quasi_newtons_method(f, x_0, f_args, **opt_args)
        case Method.COORDINATE_DESCENT:
            return zom.coordinate_descent(f, x_0, f_args, **opt_args)
        case Method.POWELLS_METHOD:
            return zom.powells_method(f, x_0, f_args, **opt_args)
        case Method.HOOKE_JEEVES_METHOD:
            return zom.hooke_jeeves_method(f, x_0, f_args, **opt_args)
        case Method.PATTERN_SEARCH:
            return zom.pattern_search(f, x_0, f_args, **opt_args)
        case Method.NELDER_MEAD_SIMPLEX:
            return zom.nelder_mead_simplex(f, f_args, x_0=x_0, **opt_args)
        case Method.DIVIDED_RECTANGLES:
            return zom.divided_rectangles(f, f_args, **opt_args)
        case Method.NOISY_DESCENT:
            return stm.noisy_descent(f, x_0, f_args, **opt_args)
        case Method.MESH_ADAPTIVE_SEARCH:
            return stm.mesh_adaptive_search(f, x_0, f_args, **opt_args)
        case Method.SIMULATED_ANNEALING:
            return stm.simulated_annealing(f, x_0, f_args, **opt_args)
        case Method.CORANA_ANNEALING:
            return stm.corana_annealing(f, x_0, f_args, **opt_args)
        case Method.CROSS_ENTROPY:
            return stm.cross_entropy(f, f_args, x_0=x_0, **opt_args)
        case Method.COVARIANCE_MATRIX_ADAPTATION:
            return stm.covariance_matrix_adaptation(f, x_0, f_args, **opt_args)
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
        case Method.HYPERRECTANGLE_CONSTRAINT:
            return con.hyperrectangle_constraint(f, x_0, f_args, **opt_args)
        case Method.PENALTY_CONSTRAINT:
            return con.penalty_constraint(f, x_0, f_args, **opt_args)
        case Method.AUGMENTED_LAGRANGE:
            return con.augmented_lagrange(f, x_0, f_args, **opt_args)
        case Method.INTERIOR_POINT_METHOD:
            return con.interior_point_method(f, x_0, f_args, **opt_args)
        case Method.NAIVE_PARETO:
            return mul.naive_pareto(f, f_args, x_0=x_0, **opt_args)
        case Method.WEIGHTED_PARETO_METHOD:
            return mul.weighted_pareto_method(f, x_0, f_args, **opt_args)
        case Method.WEIGHTED_PARETO_SCAN_METHOD:
            return mul.weighted_pareto_scan_method(f, x_0, f_args, **opt_args)
        case Method.GOAL_METHOD:
            return mul.goal_method(f, x_0, f_args, **opt_args)
        case Method.WEIGHTED_GOAL_METHOD:
            return mul.weighted_goal_method(f, x_0, f_args, **opt_args)
        case Method.WEIGHTED_GOAL_SCAN_METHOD:
            return mul.weighted_goal_scan_method(f, x_0, f_args, **opt_args)
        case Method.WEIGHTED_MIN_MAX_METHOD:
            return mul.weighted_min_max_method(f, x_0, f_args, **opt_args)
        case Method.WEIGHTED_MIN_MAX_SCAN_METHOD:
            return mul.weighted_min_max_scan_method(f, x_0, f_args, **opt_args)
        case Method.EXPONENTIAL_WEIGHT_METHOD:
            return mul.exponential_weight_method(f, x_0, f_args, **opt_args)
        case Method.EXPONENTIAL_WEIGHT_SCAN_METHOD:
            return mul.exponential_weight_scan_method(f, x_0, f_args, **opt_args)
        case Method.VECTOR_EVALUATED_GENETIC_METHOD:
            return pop.vector_evaluated_genetic_method(f, f_args, x_0=x_0, **opt_args)
        case Method.AUTO_GAUSSIAN_PROCESS:
            return gp.auto_gaussian_process(f, f_args, x_0, **opt_args)
        case _:
            return x_0
