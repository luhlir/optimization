import numpy as np
from gradientGroup import GradientGroup
from inspect import stack


def mat(x, y=None):
    if y is None:
        return x * np.swapaxes(x, 0, 1)
    else:
        return x * np.swapaxes(y, 0, 1)


class HessianGroup:

    def __init__(self, val=None, gradients=None, hessian=None, grad_group=None):
        if grad_group is not None:
            self.gradientGroup = grad_group
        else:
            self.gradientGroup = GradientGroup(val, gradients)
        self.val = self.gradientGroup.val
        self.gradients = self.gradientGroup.gradients
        if hessian is not None and not (any(stack()[i].function == "norm" for i in range(min(5, len(stack())))) and self.val == 0.0):
            self.hessian = np.array(hessian)
        else:
            self.hessian = np.zeros((self.gradients.size, self.gradients.size))

    def __str__(self):
        return "val={0}\ngradient=\n{1}\nhessian=\n{2}".format(self.val, self.gradients, self.hessian)

    def __abs__(self):
        grad = abs(self.gradientGroup)
        if self.val != 0:
            hess = np.zeros((len(self.gradients), len(self.gradients)))
            for i in range(len(self.gradients)):
                for j in range(len(self.gradients)):
                    if j < i:
                        hess[i][j] = hess[j][i]
                    else:
                        hess[i][j] = self.hessian[i][j] * abs(self.val) / self.val + self.gradients[i] * \
                                     self.gradients[j] * (abs(self.val) - 1 / abs(self.val)) / (self.val ** 2)
            return HessianGroup(grad_group=grad, hessian=hess)
        else:
            return HessianGroup(abs(self.val), self.gradients * np.inf, self.hessian * np.inf)

    def __add__(self, other):
        if not isinstance(other, HessianGroup):
            return HessianGroup(self.val + other, self.gradients, self.hessian)
        else:
            return HessianGroup(grad_group=self.gradientGroup + other.gradientGroup,
                                hessian=self.hessian + other.hessian)

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + (-1 * other)

    def __rsub__(self, other):
        return other + (-1 * self)

    def __mul__(self, other):
        if not isinstance(other, HessianGroup):
            return HessianGroup(grad_group=other * self.gradientGroup, hessian=other * self.hessian)
        else:
            hess = self.hessian * other.val + mat(self.gradients, other.gradients) + \
                other.hessian * self.val + mat(other.gradients, self.gradients)
            return HessianGroup(grad_group=other.gradientGroup * self.gradientGroup, hessian=hess)

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        return self * (other ** -1)

    def __rtruediv__(self, other):
        return other * (self ** -1)

    def __pow__(self, power, modulo=None):
        if not isinstance(power, HessianGroup):
            grad = self.gradientGroup ** power
            hess = self.hessian * power * (self.val ** (power - 1)) + mat(self.gradients) * power * \
                (power - 1) * (self.val ** (power - 2))
            return HessianGroup(grad_group=grad, hessian=hess)
        else:
            grad = self.gradientGroup ** power.gradientGroup
            if self.val <= 0:
                base = self.val + 0j
            else:
                base = self.val
            a = power.gradients * np.log(base) + self.gradients * power.val / self.val
            hess = power.hessian * np.log(base) + mat(self.gradients, power.gradients) / self.val + \
                self.hessian * power.val / self.val + mat(self.val * power.gradients + power.val * self.gradients,
                                                          self.gradients) / (self.val ** 2) + mat(a)
            return HessianGroup(grad_group=grad, hessian=hess * grad.val)

    def __rpow__(self, base, modulo=None):
        if not isinstance(base, HessianGroup):
            grad = base ** self.gradientGroup
            if base <= 0:
                base = base + 0j
            hess = self.hessian + mat(self.gradients) * np.log(base)
            return HessianGroup(grad_group=grad, hessian=hess * np.log(base) * grad.val)
        else:
            grad = base.gradientGroup ** self.gradientGroup
            if base.val <= 0:
                use_base = base.val + 0j
            else:
                use_base = base.val
            a = self.gradients * np.log(use_base) + base.gradients * self.val / base.val
            hess = self.hessian * np.log(use_base) + mat(base.gradients, self.gradients) / base.val + \
                   base.hessian * self.val / base.val + mat(base.val * self.gradients + self.val * base.gradients,
                                                             base.gradients) / (base.val ** 2) + mat(a)
            return HessianGroup(grad_group=grad, hessian=hess * grad.val)

    def __neg__(self):
        return -1 * self

    def __pos__(self):
        return 1 * self

    def __ceil__(self):
        grad = np.ceil(self.gradientGroup)
        if np.ceil(self.val) == self.val:
            return HessianGroup(grad_group=grad, hessian=self.hessian * np.inf)
        else:
            return HessianGroup(grad_group=grad, hessian=self.hessian * 0)

    def __floor__(self):
        grad = np.ceil(self.gradientGroup)
        if np.floor(self.val) == self.val:
            return HessianGroup(grad_group=grad, hessian=self.hessian * np.inf)
        else:
            return HessianGroup(grad_group=grad, hessian=self.hessian * 0)

    def __trunc__(self):
        if self < 0:
            return np.ceil(self)
        else:
            return np.floor(self)

    def __mod__(self, other):
        if self.val != other and self.val != -other:
            return HessianGroup(self.val % other, self.gradients, self.hessian)
        else:
            return HessianGroup(self.val % other, self.gradients * np.inf, self.hessian * np.inf)

    def __rmod__(self, other):
        if self.val != other and self.val != -other:
            return HessianGroup(other % self.val, self.gradients * 0)
        else:
            return HessianGroup(other % self.val, self.gradients * np.inf, self.hessian * np.inf)

    def __lt__(self, other):
        if not isinstance(other, HessianGroup):
            return self.val < other
        else:
            return self.val < other.val

    def __gt__(self, other):
        if not isinstance(other, HessianGroup):
            return self.val > other
        else:
            return self.val > other.val

    def __le__(self, other):
        if not isinstance(other, HessianGroup):
            return self.val <= other
        else:
            return self.val <= other.val

    def __ge__(self, other):
        if not isinstance(other, HessianGroup):
            return self.val >= other
        else:
            return self.val >= other.val

    def __eq__(self, other):
        if not isinstance(other, HessianGroup):
            return self.val == other
        else:
            return self.val == other.val

    def __ne__(self, other):
        if not isinstance(other, HessianGroup):
            return self.val != other
        else:
            return self.val != other.val

    def sin(self):
        return HessianGroup(grad_group=np.sin(self.gradientGroup), hessian=self.hessian * np.cos(self.val) -
                            mat(self.gradients) * np.sin(self.val))

    def cos(self):
        return HessianGroup(grad_group=np.cos(self.gradientGroup), hessian=-self.hessian * np.sin(self.val) -
                            mat(self.gradients) * np.cos(self.val))

    def tan(self):
        return HessianGroup(grad_group=np.tan(self.gradientGroup), hessian=(self.hessian * (np.cos(self.val) ** 2) + 2 *
                                                                            mat(self.gradients) *
                                                                            np.cos(self.val) * np.sin(self.val)) /
                                                                           (np.cos(self.val) ** 4))

    def sinh(self):
        return HessianGroup(grad_group=np.sinh(self.gradientGroup), hessian=self.hessian * np.cosh(self.val) +
                            mat(self.gradients) * np.sinh(self.val))

    def cosh(self):
        return HessianGroup(grad_group=np.cosh(self.gradientGroup), hessian=self.hessian * np.sinh(self.val) +
                            mat(self.gradients) * np.cosh(self.val))

    def tanh(self):
        return HessianGroup(grad_group=np.tanh(self.gradientGroup), hessian=(self.hessian - 2 * mat(self.gradients) *
                                                                             np.tanh(self.val)) *
                                                                            (1 - np.tanh(self.val) ** 2))

    def arcsin(self):
        return HessianGroup(grad_group=np.arcsin(self.gradientGroup), hessian=(self.hessian + mat(self.gradients) *
                                                                               self.val / (1 - self.val ** 2)) /
                                                                              np.sqrt(1 - self.val ** 2))

    def arccos(self):
        return HessianGroup(grad_group=np.arccos(self.gradientGroup), hessian=-(self.hessian + mat(self.gradients) *
                                                                                self.val / (1 - self.val ** 2)) /
                                                                               np.sqrt(1 - self.val ** 2))

    def arctan(self):
        return HessianGroup(grad_group=np.arctan(self.gradientGroup), hessian=-(self.hessian - 2 * mat(self.gradients) *
                                                                                self.val / (1 + self.val ** 2)) /
                                                                                (1 + self.val ** 2))

    def arcsinh(self):
        return HessianGroup(grad_group=np.arcsinh(self.gradientGroup), hessian=(self.hessian + mat(self.gradients) *
                                                                                self.val / (self.val ** 2 + 1)) /
                                                                               np.sqrt(self.val ** 2 + 1))

    def arccosh(self):
        return HessianGroup(grad_group=np.arccosh(self.gradientGroup), hessian=(self.hessian + mat(self.gradients) *
                                                                                self.val / (self.val ** 2 - 1)) /
                                                                               np.sqrt(self.val ** 2 - 1))

    def arctanh(self):
        return HessianGroup(grad_group=np.arctanh(self.gradientGroup), hessian=(self.hessian + 2 * mat(self.gradients) *
                                                                                self.val / (1 - self.val ** 2)) /
                                                                               (1 - self.val ** 2))

    def arctan2(self, other):
        if other > 0:
            return np.arctan(self / other)
        elif self > 0:
            return np.pi / 2 - np.arctan(other / self)
        elif self < 0:
            return -np.pi / 2 - np.arctan(other / self)
        elif other < 0:
            temp = np.arctan(self / other)
            if temp < 0:
                return temp + np.pi
            else:
                return temp - np.pi
        else:
            return np.arctan(self / other)

    def hypot(self, other):
        return np.sqrt(self ** 2 + other ** 2)

    def degrees(self):
        return self * 180 / np.pi

    def radians(self):
        return self * np.pi / 180

    def deg2rad(self):
        return self.radians()

    def rad2deg(self):
        return self.degrees()

    def exp(self):
        return HessianGroup(grad_group=np.exp(self.gradientGroup), hessian=np.exp(self.val) * (self.hessian +
                                                                                               mat(self.gradients)))

    def sqrt(self):
        return self ** (1 / 2)

    def log(self):
        return HessianGroup(grad_group=np.log(self.gradientGroup), hessian=(self.hessian - mat(self.gradients) /
                                                                            self.val) / self.val)

    def rint(self):
        grad = np.rint(self.gradientGroup)
        if self.val % 0.5 == 0:
            return HessianGroup(grad_group=grad, hessian=self.hessian * np.inf)
        else:
            return HessianGroup(grad_group=grad, hessian=self.hessian * 0)

    def expm1(self):
        hess = np.exp(self)
        hess.val = np.expm1(self.val)
        return hess

    def exp2(self):
        return 2 ** self

    def log10(self):
        return np.log(self) / np.log(10)

    def log2(self):
        return np.log(self) / np.log(2)

    def log1p(self):
        return np.log(1 + self)

    def sinc(self):
        return HessianGroup(grad_group=np.sinc(self.gradientGroup), hessian=self.hessian * (np.pi * self.val *
                            np.cos(np.pi * self.val) - np.sin(np.pi * self.val)) / (np.pi * self.val ** 2) +
                            mat(self.gradients) * (2 * np.sin(np.pi * self.val) - np.pi ** 2 * self.val ** 2 *
                            np.sin(np.pi * self.val) - 2 * np.pi * self.val * np.cos(np.pi * self.val)) /
                            (np.pi * self.val ** 3))

    @staticmethod
    def make_hessian_groups(values):
        groups = np.zeros(len(values), dtype=HessianGroup)
        for i in range(len(values)):
            grad = np.zeros(len(values))
            grad[i] = 1.0
            groups[i] = HessianGroup(values[i], grad)
        return groups
