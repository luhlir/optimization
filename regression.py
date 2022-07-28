import numpy as np
import itertools
import zeroOrderMethods


class Regression(object):
    """
    Abstract regression class
    """

    def __init__(self, x, y, basis_args, smoothing):
        """
        SHOULD NOT BE CALLED BY USER! ONLY BY CHILD CLASSES!
        """
        self.basis_args = basis_args
        B = np.zeros((len(x), len(self.basis_args)))
        for i in range(len(x)):
            for j in range(len(self.basis_args)):
                B[i, j] = self.basis(x[i], self.basis_args[j])
        C = np.linalg.inv(np.matmul(B.transpose(), B) + smoothing * np.identity(len(B[0])))
        self.constants = np.matmul(np.matmul(C, B.transpose()), y)
        self.train_err = 0
        for i in range(len(x)):
            self.train_err += (y[i] - self.evaluate(x[i])) ** 2
        self.train_err /= len(x)

    def basis(self, x, arg, constant=1):
        """
        Passthrough for child classes
        """
        pass

    def evaluate(self, x):
        """
        Evaluates a linear combination of basis functions

        :param x: point to evaluate at
        :return: linear combination of regression basis functions
        """
        return sum([self.basis(x, self.basis_args[i], self.constants[i]) for i in
                    range(len(self.basis_args))])

    def __call__(self, x):
        return self.evaluate(x)

    @classmethod
    def holdout_validation(cls, x, y, test_ind, **kwargs):
        """
        Calculates the holdout error of a model trained without the test_ind points

        :param x: all points used in the model
        :param y: function values of input points
        :param test_ind: list of indices for points left out of training
        :param kwargs: arguments for regression model training
        :return: the holdout error based on the given training/testing partitions
        """
        model = cls(np.delete(x, test_ind, axis=0), np.delete(y, test_ind), **kwargs)
        return np.sum((np.array(y)[test_ind] - [model(x[ind]) for ind in test_ind]) ** 2) / len(test_ind)

    @classmethod
    def cross_validation(cls, x, y, k, **kwargs):
        """
        Calculates the cross validation error using k random partitions

        :param x: all the points used in the model
        :param y: function values of input points
        :param k: number of partitions to use
        :param kwargs: arguments for regression model training
        :return: mean and std of cross validation error
        """
        left_over = len(x) % k
        indices = np.random.permutation(len(x))
        partitions = indices[:len(x) - left_over].reshape((k, int(len(x) / k))).tolist()
        i = 0
        while left_over > 0:
            partitions[i].append(indices[len(x) - left_over])
            i += 1
            left_over -= 1
        errors = [cls.holdout_validation(x, y, partition, **kwargs) / len(partition) for partition in partitions]
        return np.mean(errors)

    @classmethod
    def bootstrap_validation(cls, x, y, b, leave_one_out=False, combined=False, **kwargs):
        """
        Calculates the bootstrap validation error of the set

        :param x: all the points used in the model
        :param y: function values of input points
        :param b: number of bootstraps to generate
        :param leave_one_out: if True, only calculate errors of samples left out of models
        :param combined: if True, runs 0.632 bootstrap method
        :param kwargs: arguments for regression model training
        :return: bootstrap validation error
        """
        x = np.array(x)
        y = np.array(y)
        bootstraps = np.random.random_integers(len(x)-1, size=(b, len(x)))
        models = [cls(x[bootstrap], y[bootstrap], **kwargs) for bootstrap in bootstraps]
        loo_err = 0
        total_err = 0
        for j in range(len(x)):
            temp_loo = 0
            num_out = 0
            for i in range(len(models)):
                err = (y[j] - models[i](x[j])) ** 2
                if j not in bootstraps[i]:
                    temp_loo += err
                    num_out += 1
                total_err += err
            if num_out > 0:
                loo_err += temp_loo / num_out
        loo_err /= len(x)
        total_err /= len(x) * b
        if leave_one_out and not combined:
            return loo_err
        elif combined:
            return 0.632 * loo_err + 0.368 * total_err
        else:
            return total_err


class PolynomialRegression(Regression):
    """
    Facilitates polynomial regression using evaluate(x)
    """

    def __init__(self, x, y, degree, smoothing=0.0001):
        """
        Generates a polynomial regression model based on input points

        :param x: points to use in building regression model
        :param y: function values of input points
        :param degree: highest degree of basis functions
        :param smoothing: small float to avoid overfitting of regression model
        """
        basis_args = []
        for combo in itertools.product(range(degree + 1), repeat=len(x[0])):
            if sum(combo) <= degree:
                basis_args.append(combo)
        super().__init__(x, y, basis_args, smoothing)

    def basis(self, x, arg, constant=1):
        """
        Returns value of a single basis function based on points x and arg

        :param x: point to evaluate at
        :param arg: determines which basis function to use
        :param constant: scalar used in linear combinations
        :return: value of a basis function
        """
        if len(arg) != len(x):
            return 0
        else:
            product = 1
            for i in range(len(arg)):
                product *= x[i] ** arg[i]
            return constant * product


class SinusoidalRegression(Regression):
    """
    Facilitates sinusoidal regression using evaluate(x)
    """

    def __init__(self, x, y, degree, lower, upper, smoothing=0.0001):
        """
        Generates a sinusoidal regression model based on input points

        :param x: points to use in building regression model
        :param y: function values of input points
        :param degree: highest degree of basis functions
        :param lower: lower bound of regression space
        :param upper: upper bound of regression space
        :param smoothing: small float to avoid overfitting of regression model
        """
        self.lower = lower
        self.upper = upper
        basis_args = []
        for combo in itertools.product(range(2 * degree + 1), repeat=len(x[0])):
            summation = 0
            for power in combo:
                if power % 2 == 1:
                    summation += (power + 1) / 2
                else:
                    summation += power / 2
            if summation <= degree:
                basis_args.append(combo)
        super().__init__(x, y, basis_args, smoothing)

    def basis(self, x, arg, constant=1):
        """
        Returns value of a single basis function based on points x and arg

        :param x: point to evaluate at
        :param arg: determines which basis function to use
        :param constant: scalar used in linear combinations
        :return: value of a basis function
        """
        if len(arg) != len(x):
            return 0
        else:
            diff = np.subtract(self.upper, self.lower)
            product = 1
            for i in range(len(arg)):
                if arg[i] == 0:
                    product *= 0.5
                elif arg[i] % 2 == 1:
                    product *= np.sin(2 * np.pi * ((arg[i] + 1) / 2) * x[i] / diff[i])
                else:
                    product *= np.cos(2 * np.pi * (arg[i] / 2) * x[i] / diff[i])
            return constant * product


class RadialRegression(Regression):
    """
    Facilitates radial regression using evaluate(x)
    """

    def __init__(self, x, y, radial_func, order=2, smoothing=0.0001):
        """
        Generates a radial regression model based on input points and radial function

        :param x: points to use in building regression model
        :param y: function values of input points
        :param radial_func: radial function that takes one argument, distance from a point
        :param order: order of p-norm to use in distance calculations
        :param smoothing: small float to avoid overfitting of regression model
        """
        self.f = radial_func
        self.order = order
        super().__init__(x, y, x, smoothing)

    def basis(self, x, arg, constant=1):
        """
        Returns value of a single basis function based on points x and arg

        :param x: point to evaluate at
        :param arg: determines which basis function to use
        :param constant: scalar used in linear combinations
        :return: value of a basis function
        """
        return constant * self.f(np.linalg.norm(np.subtract(x, arg), ord=self.order))
