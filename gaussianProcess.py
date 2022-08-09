import numpy as np
import scipy.special as sp
from scipy.stats import norm

import optimization
from gradientGroup import GradientGroup
from hessianGroup import HessianGroup


class GaussianProcess(object):

    def __init__(self, x, y, mean_func="zero_mean", mean_offset=0, kernel_func="squared_exponential", noise_var=0,
                 length_scale=1, *, variance=1, power=1, bessel_order=1):
        self.x = np.array(x)
        self.y = np.array(y)
        self.y_min = np.min(self.y)
        self.x_min = self.x[np.argmin(self.y)]
        self.f_mean = getattr(self, mean_func)
        self.mean_offset = mean_offset
        self.mean = np.array([self.f_mean(x_0) for x_0 in self.x])
        self.f_kernel = getattr(self, kernel_func)
        self.variance = variance
        self.power = power
        self.length_scale = length_scale
        self.bessel_order = bessel_order
        self.covariance = np.zeros((len(self.x), len(self.x)))
        for i in range(len(self.x)):
            for j in range(len(self.x)):
                self.covariance[i][j] = self.f_kernel(self.x[i], self.x[j])
        self.covariance += noise_var * np.identity(len(self.covariance))
        self.noise_var = noise_var
        self.inverse_cov = np.linalg.inv(self.covariance + (10 ** -6) * np.identity(len(self.covariance)))
        self.theta = np.matmul(self.inverse_cov, y - self.mean).transpose()
        self.use_gradient = False

    def add_points(self, x, y):
        """
        Generates a new GaussianProcess using current and new points

        :param x: 2D array of new points
        :param y: 1D array of function values at x
        :return: new Gaussian Process
        """
        new_x = np.append(self.x, x, axis=0)
        new_y = np.append(self.y, y)
        return GaussianProcess(new_x, new_y, self.f_mean.__name__, self.mean_offset, self.f_kernel.__name__,
                               self.noise_var, self.length_scale, variance=self.variance, power=self.power,
                               bessel_order=self.bessel_order)

    def predict(self, x):
        """
        Calculates normal distribution for predicting the function value at point x

        :param x: point to evaluate at
        :return: mean and variance for normal distribution at point x
        """
        k_new_new = self.f_kernel(x, x)
        k_new_old = np.zeros((1, len(self.x)))
        k_old_new = np.zeros((len(self.x), 1))
        for i in range(len(self.x)):
            k_new_old[0, i] = self.f_kernel(x, self.x[i])
            k_old_new[i, 0] = self.f_kernel(self.x[i], x)
        if self.use_gradient:
            for i in range(len(self.x)):
                gradient = GradientGroup.make_gradient_groups(self.x[i])
                gradient_new = self.f_kernel(x, gradient)
                gradient_old = self.f_kernel(gradient, x)
                k_new_old = np.append(k_new_old, gradient_new.gradients, axis=1)
                k_old_new = np.append(k_old_new, gradient_old.gradients, axis=0)
        mean = self.f_mean(x) + np.matmul(self.theta, k_old_new)
        var = k_new_new - np.matmul(np.matmul(k_new_old, self.inverse_cov), k_old_new)
        return np.squeeze(mean), abs(np.squeeze(var))

    # TODO: Fix this gradient stuff haha. I think without gradient information, this is fine
    def add_gradient(self, y_prime):
        """
        Broken, in need of loving hands

        :param y_prime: list of gradients at previously provided design points
        :return: None
        """
        self.use_gradient = True
        self.y = np.append(self.y, y_prime)
        left_grad = np.zeros((len(self.x)*len(self.x[0]), len(self.x)))
        right_grad = np.zeros((len(self.x), len(self.x)*len(self.x[0])))
        hess = np.zeros((len(self.x)*len(self.x[0]), len(self.x)*len(self.x[0])))
        mean = np.zeros(len(self.x) * len(self.x[0]))
        for i in range(len(self.x)):
            mean[i * len(self.x[i]):(i + 1) * len(self.x[i])] = self.f_mean(GradientGroup.make_gradient_groups(self.x[i])).gradients
            for j in range(len(self.x)):
                if i != j:
                    hessians = HessianGroup.make_hessian_groups(np.append(self.x[i], self.x[j]))
                    output = self.f_kernel(hessians[:len(self.x[i])], hessians[len(self.x[i]):])
                    left_grad[i * len(self.x[0]):(i + 1) * len(self.x[0]), j] = output.gradients[:len(self.x[i])]
                    right_grad[i, j * len(self.x[0]):(j + 1) * len(self.x[0])] = output.gradients[len(self.x[j]):]
                    hess[i * len(self.x[0]):(i + 1) * len(self.x[0]),
                    j * len(self.x[0]):(j + 1) * len(self.x[0])] = output.hessian[len(self.x[0]):, :len(self.x[0])]
                else:
                    hessians = HessianGroup.make_hessian_groups(self.x[i])
                    output = self.f_kernel(hessians, hessians)
                    left_grad[i * len(self.x[0]):(i + 1) * len(self.x[0]), j] = output.gradients
                    right_grad[i, j * len(self.x[0]):(j + 1) * len(self.x[0])] = output.gradients
                    hess[i * len(self.x[0]):(i + 1) * len(self.x[0]),
                    j * len(self.x[0]):(j + 1) * len(self.x[0])] = output.hessian
        new_cov = np.zeros((len(self.x) * (1 + len(self.x[0])), len(self.x) * (1 + len(self.x[0]))))
        new_cov[:len(self.x), :len(self.x)] = self.covariance
        new_cov[len(self.x):, :len(self.x)] = left_grad
        new_cov[:len(self.x), len(self.x):] = right_grad
        new_cov[len(self.x):, len(self.x):] = hess
        self.covariance = new_cov
        self.inverse_cov = np.linalg.inv(self.covariance + (10 ** -6) * np.identity(len(self.covariance)))
        self.mean = np.append(self.mean, mean)
        self.theta = np.matmul(self.inverse_cov, self.y - self.mean).transpose()

    def __call__(self, x):
        return self.predict(x)

    def zero_mean(self, x):
        return x[0] * 0 + self.mean_offset

    def constant(self, x, x_0):
        return self.variance + 0 * x[0] + 0 * x_0[0]

    def linear(self, x, x_0):
        return np.sum(self.variance * np.multiply(x, x_0))

    def polynomial(self, x, x_0):
        return (np.dot(x, x_0) + self.variance) ** self.power

    def exponential(self, x, x_0):
        return np.exp(-np.linalg.norm(np.subtract(x, x_0)) / self.length_scale)

    def gamma_exponential(self, x, x_0):
        return np.exp(-(np.linalg.norm(np.subtract(x, x_0)) / self.length_scale) ** self.power)

    def squared_exponential(self, x, x_0):
        return np.exp(-(np.linalg.norm(np.subtract(x, x_0)) ** 2) / (2 * self.length_scale ** 2))

    def matern(self, x, x_0):
        temp = np.sqrt(2 * self.bessel_order) * np.linalg.norm(x, x_0) / self.length_scale
        return ((2 ** (1 - self.bessel_order)) * (temp ** self.bessel_order) *
                sp.yv(self.bessel_order, temp)) / sp.gamma(self.bessel_order)

    def rational_quadratic(self, x, x_0):
        return (1 + (np.linalg.norm(x, x_0) ** 2) / (2 * self.power * self.length_scale ** 2)) ** -self.power

    def neural_network(self, x, x_0):
        x_bar = np.ones(len(x) + 1)
        x_0_bar = np.ones(len(x_0) + 1)
        x_bar[1:] = x
        x_0_bar[1:] = x_0
        numerator = 2 * np.matmul(np.matmul(np.transpose(x_bar), self.variance), x_0_bar)
        denominator_0 = 1 + 2 * np.matmul(np.matmul(np.transpose(x_bar), self.variance), x_bar)
        denominator_1 = 1 + 2 * np.matmul(np.matmul(np.transpose(x_0_bar), self.variance), x_0_bar)
        return np.arcsin(numerator / np.sqrt(denominator_0 * denominator_1))

    def log_likelihood(self):
        """
        Calculates the log-likelihood of the current Gaussian Process

        :return:
        """
        first_part = len(self.x[0]) * np.log(2 * np.pi)
        second_part = np.log(np.linalg.norm(self.covariance))
        third_part = np.matmul(self.theta, self.y - self.mean)
        return -(first_part + second_part + third_part) / 2

    def fit_hyperparameters(self):
        """
        Performs gradient descent to maximize the log-likelihood estimate

        :return: A new, fitted Gaussian Process
        """
        x_0 = [self.length_scale, self.variance, self.power, self.bessel_order]
        x_fit = optimization.optimize(GaussianProcess.fit_log_likelihood, x_0, {"gp": self},
                                      f_prime=GaussianProcess.fit_log_likelihood_gradient, fp_args={"gp": self},
                                      auto_diff=False, lin_args={"f_prime": GaussianProcess.fit_log_likelihood_gradient,
                                                                 "fp_args": {"gp": self}, "auto_diff": False})
        length_scale, variance, power, bessel_order = x_fit
        print("New hyperparameters: \n\tlength_scale=" + str(length_scale) + "\n\tvariance=" + str(variance) +
              "\n\tpower=" + str(power) + "\n\tbessel_order=" + str(bessel_order))
        gp = GaussianProcess(self.x, self.y, self.f_mean.__name__, self.mean_offset, self.f_kernel.__name__,
                             self.noise_var, length_scale, variance=variance, power=power, bessel_order=bessel_order)
        print("Improvements: \n\told log-likelihood=" + str(self.log_likelihood()) +
              "\n\tnew log-likelihood=" + str(gp.log_likelihood()))
        return gp

    def prediction_based(self, x):
        """
        Calculates the mean of the process at x for prediction-based exploration

        :param x: point to evaluate at
        :return: mean of the process at x
        """
        mean, var = self.predict(x)
        return mean

    def error_based(self, x, *, invert=False):
        """
        returns the variance at x for error-based exploration

        :param x: point to evaluate at
        :param invert: invert the return value for maximization
        :return: variance of the process at x
        """
        mean, var = self.predict(x)
        if invert:
            return -var
        return var

    def lower_confidence(self, x, alpha):
        """
        Calculates a combination of mean and variance of the process at x for lower confidence bound exploration

        :param x: point to evaluate at
        :param alpha: nonnegative scalar for preferencing variance
        :return: combination of mean and variance at x
        """
        mean, var = self.predict(x)
        return mean - alpha * var

    def improvement_prob(self, x, *, invert=False):
        """
        Calculates the probability of improving the lowest known function value at point x

        :param x: point to evaluate at
        :param invert: invert the return value for maximization
        :return: probability of improving at point x
        """
        mean, var = self.predict(x)
        prob = norm.cdf((self.y_min - mean) / np.sqrt(var))
        if invert:
            return -prob
        return prob

    def expected_improv(self, x, *, invert=False):
        """
        Calculates the expected improvement on the lowest known function value at point x

        :param x: point to evaluate at
        :param invert: invert the return value for maximization
        :return: expected improvement at point x
        """
        mean, var = self.predict(x)
        improv = (self.y_min - mean) * self.improvement_prob(x) + var * norm.pdf(self.y_min, loc=mean, scale=np.sqrt(var))
        if invert:
            return -improv
        return improv

    def explore_point(self, method="error_based", lower=None, upper=None, *, alpha=0.5):
        """
        Finds a likely best point to evaluate based on exploration method within the given interval

        :param method: exploration method
        :param lower: lower bound for the exploration
        :param upper: upper bound for the exploration
        :param alpha: scalar for lower confidence bound exploration
        :return: likely best point to explore next
        """
        f = getattr(self, method)
        if lower is None:
            lower = np.min(self.x, axis=0)
        if upper is None:
            upper = np.max(self.x, axis=0)
        if method == "lower_confidence":
            f_args = {"alpha": alpha}
        elif method == "error_based" or method == "improvement_prob" or method == "expected_improv":
            f_args = {"invert": True}
        else:
            f_args = {}
        return optimization.optimize(f, None, f_args=f_args, opt_method="divided_rectangles", interval_low=lower, interval_high=upper)

    @staticmethod
    def fit_log_likelihood(kernel_vars, gp):
        """
        Inverts and updates the log-likelihood of the Gaussian Process for fitting hyperparameters and finding the maximum likelihood estimate

        :param kernel_vars: array of kernel variables (length_scale, variance, power, bessel_order)
        :param gp: gaussian process to update
        :return: inverted and updated log-likelihood
        """
        length_scale, variance, power, bessel_order = kernel_vars
        new_gaussian = GaussianProcess(gp.x, gp.y, gp.f_mean.__name__, gp.mean_offset, gp.f_kernel.__name__, gp.noise_var,
                                       length_scale, variance=variance, power=power, bessel_order=bessel_order)
        return -new_gaussian.log_likelihood()

    @staticmethod
    def fit_log_likelihood_gradient(kernel_vars, gp):
        """
        Calculates the inverted gradient of the log-likelihood of the given Gaussian Process with updated kernel vars (assumes a zero-mean

        :param kernel_vars: array of kernel variables (length_scale, variance, power, bessel_order)
        :param gp: gaussian process to update
        :return: inverted and updated gradient of the log-likelihood
        """
        length_scale, variance, power, bessel_order = kernel_vars
        new_gaussian = GaussianProcess(gp.x, gp.y, gp.f_mean.__name__, gp.mean_offset, gp.f_kernel.__name__, gp.noise_var,
                                       length_scale, variance=variance, power=power, bessel_order=bessel_order)
        clean_inv = np.linalg.inv(new_gaussian.covariance - new_gaussian.noise_var * np.identity(len(new_gaussian.covariance)))

        # Do the gradient calculations
        new_gaussian.length_scale, new_gaussian.variance, new_gaussian.power, new_gaussian.bessel_order =\
            GradientGroup.make_gradient_groups(kernel_vars)
        d_K = np.zeros(new_gaussian.covariance.shape, dtype=GradientGroup)
        for i in range(len(new_gaussian.x)):
            for j in range(len(new_gaussian.x)):
                d_K[i][j] = new_gaussian.f_kernel(new_gaussian.x[i], new_gaussian.x[j])

        # Separate the gradients and do the final calculations
        pre = np.matmul(new_gaussian.y.transpose(), clean_inv)
        post = np.matmul(clean_inv, new_gaussian.y)
        gradients = np.zeros(len(kernel_vars))
        for i in range(len(kernel_vars)):
            sel_grad = [[d_K[j][k].gradients[i, 0] for k in range(len(d_K))] for j in range(len(d_K))]
            trace = np.trace(np.matmul(new_gaussian.inverse_cov, sel_grad))
            gradients[i] = trace - np.matmul(np.matmul(pre, sel_grad), post)

        return gradients / 2
