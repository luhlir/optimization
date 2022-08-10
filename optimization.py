import firstOrderMethods as fom
import secondOrderMethods as som
import zeroOrderMethods as zom
import stochasticMethods as stm
import populationMethods as pop
import constraintMethods as con
import multiobjectiveMethods as mul
import gaussianProcess as gp


def optimize(f, x_0, f_args={}, opt_method="gradient_descent", **opt_args):

    if opt_method == "gradient_descent":
        return fom.gradient_descent(f, x_0, f_args, **opt_args)
    elif opt_method == "conjugate_descent":
        return fom.conjugate_descent(f, x_0, f_args, **opt_args)
    elif opt_method == "momentum_descent":
        return fom.momentum_descent(f, x_0, f_args, **opt_args)
    elif opt_method == "nesterov_descent":
        return fom.momentum_descent(f, x_0, f_args, **opt_args)
    elif opt_method == "adagrad_descent":
        return fom.momentum_descent(f, x_0, f_args, **opt_args)
    elif opt_method == "rmsprop_descent":
        return fom.momentum_descent(f, x_0, f_args, **opt_args)
    elif opt_method == "adadelta_descent":
        return fom.momentum_descent(f, x_0, f_args, **opt_args)
    elif opt_method == "adam_descent":
        return fom.momentum_descent(f, x_0, f_args, **opt_args)
    elif opt_method == "hypergradient_descent":
        return fom.momentum_descent(f, x_0, f_args, **opt_args)
    elif opt_method == "newtons_method":
        return som.newtons_method(f, x_0, f_args, **opt_args)
    elif opt_method == "quasi_newtons_method":
        return som.quasi_newtons_method(f, x_0, f_args, **opt_args)
    elif opt_method == "coordinate_descent":
        return zom.coordinate_descent(f, x_0, f_args, **opt_args)
    elif opt_method == "powells_method":
        return zom.powells_method(f, x_0, f_args, **opt_args)
    elif opt_method == "hooke_jeeves_method":
        return zom.hooke_jeeves_method(f, x_0, f_args, **opt_args)
    elif opt_method == "pattern_search":
        return zom.pattern_search(f, x_0, f_args, **opt_args)
    elif opt_method == "nelder_mead_simplex":
        return zom.nelder_mead_simplex(f, f_args, x_0=x_0, **opt_args)
    elif opt_method == "divided_rectangles":
        return zom.divided_rectangles(f, f_args, **opt_args)
    elif opt_method == "noisy_descent":
        return stm.noisy_descent(f, x_0, f_args, **opt_args)
    elif opt_method == "mesh_adaptive_search":
        return stm.mesh_adaptive_search(f, x_0, f_args, **opt_args)
    elif opt_method == "simulated_annealing":
        return stm.simulated_annealing(f, x_0, f_args, **opt_args)
    elif opt_method == "corana_annealing":
        return stm.corana_annealing(f, x_0, f_args, **opt_args)
    elif opt_method == "cross_entropy":
        return stm.cross_entropy(f, f_args, x_0=x_0, **opt_args)
    elif opt_method == "covariance_matrix_adaptation":
        return stm.covariance_matrix_adaptation(f, x_0, f_args, **opt_args)
    elif opt_method == "genetic_method":
        return pop.genetic_method(f, f_args, x_0=x_0, **opt_args)
    elif opt_method == "differential_evolution":
        return pop.differential_evolution(f, f_args, x_0=x_0, **opt_args)
    elif opt_method == "particle_swarm":
        return pop.particle_swarm(f, f_args, x_0=x_0, **opt_args)
    elif opt_method == "firefly_method":
        return pop.firefly_method(f, f_args, x_0=x_0, **opt_args)
    elif opt_method == "cuckoo_search":
        return pop.cuckoo_search(f, f_args, x_0=x_0, **opt_args)
    elif opt_method == "hyperrectangle_contraint":
        return con.hyperrectangle_constraint(f, x_0, f_args, **opt_args)
    elif opt_method == "penalty_constraint":
        return con.penalty_constraint(f, x_0, f_args, **opt_args)
    elif opt_method == "augmented_lagrange":
        return con.augmented_lagrange(f, x_0, f_args, **opt_args)
    elif opt_method == "interior_point_method":
        return con.interior_point_method(f, x_0, f_args, **opt_args)
    elif opt_method == "naive_pareto":
        return mul.naive_pareto(f, f_args, x_0=x_0, **opt_args)
    elif opt_method == "weighted_pareto_method":
        return mul.weighted_pareto_method(f, x_0, f_args, **opt_args)
    elif opt_method == "weighted_pareto_scan_method":
        return mul.weighted_pareto_scan_method(f, x_0, f_args, **opt_args)
    elif opt_method == "goal_method":
        return mul.goal_method(f, x_0, f_args, **opt_args)
    elif opt_method == "weighted_goal_method":
        return mul.weighted_goal_method(f, x_0, f_args, **opt_args)
    elif opt_method == "weighted_goal_scan_method":
        return mul.weighted_goal_scan_method(f, x_0, f_args, **opt_args)
    elif opt_method == "weighted_min_max_method":
        return mul.weighted_min_max_method(f, x_0, f_args, **opt_args)
    elif opt_method == "weighted_min_max_scan_method":
        return mul.weighted_min_max_scan_method(f, x_0, f_args, **opt_args)
    elif opt_method == "exponential_weight_method":
        return mul.exponential_weight_method(f, x_0, f_args, **opt_args)
    elif opt_method == "exponential_weight_scan_method":
        return mul.exponential_weight_scan_method(f, x_0, f_args, **opt_args)
    elif opt_method == "vector_evaluated_genetic_method":
        return pop.vector_evaluated_genetic_method(f, f_args, x_0=x_0, **opt_args)
    elif opt_method == "auto_gaussian_process":
        return gp.auto_gaussian_process(f, f_args, x_0, **opt_args)
    return x_0
