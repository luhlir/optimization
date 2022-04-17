import numpy as np


class GradientGroup:

    def __init__(self, val, gradients):
        self.val = val * 1.0
        self.gradients = np.array(gradients) * 1.0
        self.gradients = self.gradients.reshape((self.gradients.size, 1))


    def __str__(self):
        return "val={0}\ngradient=({1})".format(self.val, self.gradients)

    def __abs__(self):
        if self.val != 0:
            return GradientGroup(abs(self.val), self.gradients * (self.val / abs(self.val)))
        else:
            return GradientGroup(abs(self.val), self.gradients * np.inf)

    def __add__(self, other):
        if not isinstance(other, GradientGroup):
            return GradientGroup(self.val + other, self.gradients)
        else:
            return GradientGroup(self.val + other.val, self.gradients + other.gradients)

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + (-1 * other)

    def __rsub__(self, other):
        return other + (-1 * self)

    def __mul__(self, other):
        if not isinstance(other, GradientGroup):
            return GradientGroup(other * self.val, other * self.gradients)
        else:
            return GradientGroup(self.val * other.val, self.gradients * other.val + other.gradients * self.val)

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        return self * (1 / other)

    def __rtruediv__(self, other):
        return other * (self ** -1)

    def __pow__(self, power, modulo=None):
        if not isinstance(power, GradientGroup):
            return GradientGroup(self.val ** power, self.gradients * (power * (self.val ** (power - 1))))
        else:
            new_val = self.val ** power.val
            others = power * np.log(self)
            return GradientGroup(new_val, new_val * others.gradients)

    def __rpow__(self, base, modulo=None):
        if not isinstance(base, GradientGroup):
            if base <= 0:
                base = base + 0j
            return GradientGroup(base ** self.val, self.gradients * base ** self.val * np.log(base))
        else:
            new_val = base.val ** self.val
            others = self * np.log(base)
            return GradientGroup(new_val, new_val * others.gradients)

    def __neg__(self):
        return -1 * self

    def __pos__(self):
        return 1 * self

    def __ceil__(self):
        if np.ceil(self.val) == self.val:
            return GradientGroup(np.ceil(self.val), self.gradients * np.inf)
        else:
            return GradientGroup(np.ceil(self.val), self.gradients * 0)

    def __floor__(self):
        if np.floor(self.val) == self.val:
            return GradientGroup(np.floor(self.val), self.gradients * np.inf)
        else:
            return GradientGroup(np.floor(self.val), self.gradients * 0)

    def __trunc__(self):
        if self < 0:
            return np.ceil(self)
        else:
            return np.floor(self)

    def __mod__(self, other):
        if self.val != other and self.val != -other:
            return GradientGroup(self.val % other, self.gradients)
        else:
            return GradientGroup(self.val % other, self.gradients * np.inf)

    def __rmod__(self, other):
        if self.val != other and self.val != -other:
            return GradientGroup(other % self.val, self.gradients * 0)
        else:
            return GradientGroup(other % self.val, self.gradients * np.inf)

    def __lt__(self, other):
        if not isinstance(other, GradientGroup):
            return self.val < other
        else:
            return self.val < other.val

    def __gt__(self, other):
        if not isinstance(other, GradientGroup):
            return self.val > other
        else:
            return self.val > other.val

    def __le__(self, other):
        if not isinstance(other, GradientGroup):
            return self.val <= other
        else:
            return self.val <= other.val

    def __ge__(self, other):
        if not isinstance(other, GradientGroup):
            return self.val >= other
        else:
            return self.val >= other.val

    def __eq__(self, other):
        if not isinstance(other, GradientGroup):
            return self.val == other
        else:
            return self.val == other.val

    def __ne__(self, other):
        if not isinstance(other, GradientGroup):
            return self.val != other
        else:
            return self.val != other.val

    def sin(self):
        return GradientGroup(np.sin(self.val), self.gradients * np.cos(self.val))

    def cos(self):
        return GradientGroup(np.cos(self.val), -1 * self.gradients * np.sin(self.val))

    def tan(self):
        return GradientGroup(np.tan(self.val), self.gradients / (np.cos(self.val) ** 2))

    def sinh(self):
        return GradientGroup(np.sinh(self.val), self.gradients * np.cosh(self.val))

    def cosh(self):
        return GradientGroup(np.cosh(self.val), self.gradients * np.sinh(self.val))

    def tanh(self):
        return GradientGroup(np.tanh(self.val), self.gradients * (1 - np.tanh(self.val) ** 2))

    def arcsin(self):
        return GradientGroup(np.arcsin(self.val), self.gradients / np.sqrt(1 - (self.val ** 2)))

    def arccos(self):
        return GradientGroup(np.arccos(self.val), -1 * self.gradients / np.sqrt(1 - (self.val ** 2)))

    def arctan(self):
        return GradientGroup(np.arctan(self.val), self.gradients / (1 + (self.val ** 2)))

    def arcsinh(self):
        return GradientGroup(np.arcsinh(self.val), self.gradients / np.sqrt((self.val ** 2) + 1))

    def arccosh(self):
        return GradientGroup(np.arccosh(self.val), self.gradients / np.sqrt((self.val ** 2) - 1))

    def arctanh(self):
        return GradientGroup(np.arctanh(self.val), self.gradients / (1 - (self.val ** 2)))

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
        return GradientGroup(np.exp(self.val), self.gradients * np.exp(self.val))

    def sqrt(self):
        return self ** (1 / 2)

    def log(self):
        if self.val <= 0:
            base = self.val + 0j
        else:
            base = self.val
        return GradientGroup(np.log(base), self.gradients / self.val)

    def rint(self):
        if self.val % 0.5 == 0:
            return GradientGroup(np.rint(self.val), self.gradients * np.inf)
        else:
            return GradientGroup(np.rint(self.val), self.gradients * 0)

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
        return GradientGroup(np.sinc(self.val), self.gradients * (np.pi * self.val * np.cos(np.pi * self.val) -
                                                                  np.sin(np.pi * self.val)) / (np.pi * self.val ** 2))

    @staticmethod
    def make_gradient_groups(values):
        groups = np.zeros(len(values), dtype=GradientGroup)
        for i in range(len(values)):
            grad = np.zeros(len(values))
            grad[i] = 1.0
            groups[i] = GradientGroup(values[i], grad)
        return groups
