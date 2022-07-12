import numpy as np
import itertools
import zeroOrderMethods


class Regression(object):

    def __init__(self, f, f_args, in_points, basis_args, smoothing, multithreaded):
        self.basis_args = basis_args
        B = np.zeros((len(in_points), len(self.basis_args)))
        for i in range(len(in_points)):
            for j in range(len(self.basis_args)):
                B[i, j] = self.basis(in_points[i], self.basis_args[j])
        y = zeroOrderMethods.batch_eval(f, in_points, f_args, multithreaded)
        C = np.linalg.inv(np.matmul(B.transpose(), B) + smoothing * np.identity(len(B[0])))
        self.constants = np.matmul(np.matmul(C, B.transpose()), y)

    def basis(self, x, arg, constant=1):
        pass

    def evaluate(self, x):
        return sum([self.basis(x, self.basis_args[i], self.constants[i]) for i in
                    range(len(self.basis_args))])


class PolynomialRegression(Regression):

    def __init__(self, f, in_points, degree, f_args={}, smoothing=0.0001, multithreaded=False):
        basis_args = []
        for combo in itertools.product(range(degree + 1), repeat=len(in_points[0])):
            if sum(combo) <= degree:
                basis_args.append(combo)
        super().__init__(f, f_args, in_points, basis_args, smoothing, multithreaded)

    def basis(self, x, arg, constant=1):
        if len(arg) != len(x):
            return 0
        else:
            product = 1
            for i in range(len(arg)):
                product *= x[i] ** arg[i]
            return constant * product


class SinusoidalRegression(Regression):

    def __init__(self, f, in_points, degree, lower, upper, f_args={}, smoothing=0.0001, multithreaded=False):
        self.lower = lower
        self.upper = upper
        basis_args = []
        for combo in itertools.product(range(2 * degree + 1), repeat=len(in_points[0])):
            summation = 0
            for power in combo:
                if power % 2 == 1:
                    summation += (power + 1) / 2
                else:
                    summation += power / 2
            if summation <= degree:
                basis_args.append(combo)
        super().__init__(f, f_args, in_points, basis_args, smoothing, multithreaded)

    def basis(self, x, arg, constant=1):
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

    def __init__(self, f, in_points, radial_func, order, f_args={}, smoothing=0.0001, multithreaded=False):
        self.f = radial_func
        self.order = order
        super().__init__(f, f_args, in_points, in_points, smoothing, multithreaded)

    def basis(self, x, arg, constant=1):
        return constant * self.f(np.linalg.norm(np.subtract(x, arg), ord=self.order))
