import firstOrderMethods as fom
import secondOrderMethods as som
import zeroOrderMethods as zom


def optimize(f, x_0, f_args=None, opt_method="newtons_method", opt_args=None):
    if f_args is None:
        f_args = {}
    if opt_args is None:
        opt_args = {}

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
    return x_0
