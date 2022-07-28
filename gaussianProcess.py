import numpy as np
import scipy.special as sp
from gradientGroup import GradientGroup
from hessianGroup import HessianGroup


class GaussianProcess(object):

    def __init__(self, x, y, mean_func="zero_mean", mean_offset=0, kernel_func="squared_exponential", length_scale=1,
                 noise_var=0, *, variance=1, power=None, bessel_order=None):
        self.x = np.array(x)
        self.y = np.array(y)
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

    def predict(self, x):
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
        return np.squeeze(mean), np.squeeze(var)

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
        return self.variance

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
