import sampleFunctions
import sampleFunctions as sample
import samplingPlans
import samplingPlans as plan
from gradientGroup import GradientGroup
import zeroOrderMethods
from optimization import optimize
import matplotlib.pyplot as plt
import multiobjectiveMethods
from sklearn.mixture import GaussianMixture
from zeroOrderMethods import make_minimal_positive_spanning_set
from gaussianProcess import GaussianProcess
import numpy as np
import multiprocessing
import time
import regression
import itertools


def radial_func(r):
    return np.exp(-2 * (r ** 2))


def sin_wrapper(x):
    return np.sin(x[0]) ** 2


if __name__ == "__main__":

    degree = 2
    smoothing = 0.01
    noise = 0.01
    scale = 0.2
    p = 2
    lower, upper = 0, 10
    test_point_count = 100
    # lower, upper = [-1, -1], [2, 2]
    # points = samplingPlans.full_factorial_samples(lower, upper, 4, True)
    # point_vals = zeroOrderMethods.batch_eval(sample.rosenbrock, points, {}, False)
    points = samplingPlans.full_factorial_samples([lower], [upper], 7, True)
    point_vals = np.add(zeroOrderMethods.batch_eval(sin_wrapper, points, {}, False), np.random.normal(0, noise, 7))
    test_points = np.linspace(lower, upper, num=test_point_count).reshape((test_point_count, 1))
    gp = GaussianProcess(points, point_vals, kernel_func="exponential", length_scale=scale, noise_var=noise)
    gp = gp.fit_hyperparameters()
    i = 0
    while i < 40:
        next_point = gp.explore_point(method="expected_improv", alpha=0.5, lower=[lower], upper=[upper])
        print(next_point)
        next_val = sin_wrapper(next_point) + np.random.normal(0, noise)
        gp = gp.add_points([next_point], [next_val])
        i += 1
    nongrad_guess = [gp(test_points[i]) for i in range(len(test_points))]
    nongrad_mean = [nongrad_guess[i][0] for i in range(len(nongrad_guess))]
    nongrad_var = [nongrad_guess[i][1] for i in range(len(nongrad_guess))]
    nongrad_upper = np.add(nongrad_mean, 1.96 * np.sqrt(nongrad_var))
    nongrad_lower = np.subtract(nongrad_mean, 1.96 * np.sqrt(nongrad_var))

    # gp = gp.fit_hyperparameters()
    # grad_guess = [gp(test_points[i]) for i in range(len(test_points))]
    # grad_mean = [grad_guess[i][0] for i in range(len(grad_guess))]
    # grad_var = [grad_guess[i][1] for i in range(len(grad_guess))]
    # grad_upper = np.add(grad_mean, 1.96 * np.sqrt(grad_var))
    # grad_lower = np.subtract(grad_mean, 1.96 * np.sqrt(grad_var))
    actual = [sin_wrapper(test_points[i]) + np.random.normal(0, noise) for i in range(len(test_points))]
    plt.plot(test_points.flatten(), nongrad_lower, 'red')
    plt.plot(test_points.flatten(), nongrad_upper, 'red')
    # plt.plot(test_points.flatten(), grad_lower, 'green')
    # plt.plot(test_points.flatten(), grad_upper, 'green')
    plt.plot(test_points.flatten(), actual, 'black')
    plt.show()
    # print("Actual: " + str(sin_wrapper(test_point)))
    # mean, var = gp(test_point)
    # print("Estimated: " + str(mean) + " (95% conf: [" + str(mean - 1.96 * np.sqrt(var)) + ", " +
    #       str(mean + 1.96 * np.sqrt(var)) + "])")
    exit(0)

    basis_args = []

    print(regression.PolynomialRegression.bootstrap_validation(points, point_vals, 7, leave_one_out=False, combined=False,
                                                         degree=degree, smoothing=smoothing))
    print(regression.PolynomialRegression.bootstrap_validation(points, point_vals, 7, leave_one_out=True, combined=False,
                                                         degree=degree, smoothing=smoothing))
    print(regression.PolynomialRegression.bootstrap_validation(points, point_vals, 7, leave_one_out=False, combined=True,
                                                         degree=degree, smoothing=smoothing))
    polyreg = regression.PolynomialRegression(points, point_vals, degree, smoothing=smoothing)

    # print(regression.SinusoidalRegression.bootstrap_validation(points, point_vals, 7, leave_one_out=False, combined=False,
    #                                                      degree=degree, lower=lower, upper=upper, smoothing=smoothing))
    # print(regression.SinusoidalRegression.bootstrap_validation(points, point_vals, 7, leave_one_out=True, combined=False,
    #                                                      degree=degree, lower=lower, upper=upper, smoothing=smoothing))
    # print(regression.SinusoidalRegression.bootstrap_validation(points, point_vals, 7, leave_one_out=False, combined=True,
    #                                                      degree=degree, lower=lower, upper=upper, smoothing=smoothing))
    # polyreg = regression.SinusoidalRegression(points, point_vals, degree, lower, upper, smoothing=smoothing)

    # print(regression.RadialRegression.bootstrap_validation(points, point_vals, 7, leave_one_out=False, combined=False,
    #                                                      radial_func=radial_func, smoothing=smoothing))
    # print(regression.RadialRegression.bootstrap_validation(points, point_vals, 7, leave_one_out=True, combined=False,
    #                                                      radial_func=radial_func, smoothing=smoothing))
    # print(regression.RadialRegression.bootstrap_validation(points, point_vals, 7, leave_one_out=False, combined=True,
    #                                                      radial_func=radial_func, smoothing=smoothing))
    # polyreg = regression.RadialRegression(points, point_vals, radial_func, smoothing=smoothing)
    print(polyreg.train_err)

    x = np.linspace(lower[0], upper[0], 100)
    y = np.linspace(lower[1], upper[1], 100)
    tst = []
    z = []
    z_true = []
    for xy in itertools.product(x, y):
        tst.append([xy[0], xy[1]])
        # z.append(sum([sinusoidal_basis(xy, basis_args[i], lower, upper, constants[i]) for i in range(len(basis_args))]))
        # z.append(sum([polynomial_basis(xy, basis_args[i], constants[i]) for i in range(len(basis_args))]))
        z.append(polyreg.evaluate(xy))
        z_true.append(sample.rosenbrock(xy))
    tst = np.array(tst)
    fig, ax = plt.subplots(2)
    ax[0].scatter(tst[:,1], tst[:,0], 10, z)
    ax[1].scatter(tst[:,1], tst[:,0], 10, z_true)
    plt.show()
