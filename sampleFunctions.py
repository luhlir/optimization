import numpy as np


def list_or_ndarray(x):
    return isinstance(x, list) or isinstance(x, np.ndarray)


# len(x) dimensions, global min of 0 at origin, many local minima
def ackley(x, a=20, b=0.2, c=2 * np.pi):
    if not list_or_ndarray(x):
        return np.inf
    d = len(x)
    if isinstance(x, np.ndarray):
        return -a * np.exp(-b * np.sqrt(sum(x ** 2) / d)) - np.exp(sum(np.cos(c * x)) / d) + a + np.exp(1)
    else:
        sum_squ = 0
        sum_cos = 0
        for val in x:
            sum_squ += val ** 2
            sum_cos += np.cos(c * val)
        return -a * np.exp(-b * np.sqrt(sum_squ / d)) - np.exp(sum_cos / d) + a + np.exp(1)


# 2 dimensions, quadratic, global minimum of 0 at (1,3)
def booth(x=None, y=None, z=None):
    if x is not None and list_or_ndarray(x) and len(x) == 2:
        return (x[0] + 2 * x[1] - 7) ** 2 + (2 * x[0] + x[1] - 5) ** 2
    elif (y is not None and not list_or_ndarray(y)) and (z is not None and not list_or_ndarray(z)):
        return (y + 2 * z - 7) ** 2 + (2 * y + z - 5) ** 2
    else:
        return np.inf


# 2 dimensions, only global minima (see pg 428 of Kochenderfer & Wheeler)
def branin(x=None, y=None, z=None, a=1, b=5.1/(4*(np.pi**2)), c=5/np.pi, r=6, s=10, t=1/(8*np.pi)):
    if x is not None and list_or_ndarray(x) and len(x) == 2:
        return a * (x[1] - b * x[0] ** 2 + c * x[0] - r) ** 2 + s * (1 - t) * np.cos(x[0]) + s
    elif (y is not None and not list_or_ndarray(y)) and (z is not None and not list_or_ndarray(z)):
        return a * (z - b * y ** 2 + c * y - r) ** 2 + s * (1 - t) * np.cos(y) + s
    else:
        return np.inf


# 2 dimensions, no global minimum (arctan at (0,0))
def flower(x=None, y=None, z=None, a=1, b=1, c=4):
    if x is not None and list_or_ndarray(x) and len(x) == 2:
        return a * np.linalg.norm(x) + b * np.sin(c * np.arctan2(x[1], x[0]))
    elif (y is not None and not list_or_ndarray(y)) and (z is not None and not list_or_ndarray(z)):
        return a * np.linalg.norm([y, z]) + b * np.sin(c * np.arctan2(z, y))
    else:
        return np.inf


# len(x) dimensions, steep valleys, global minimum depends on dimensions (see pg 430 of Kochenderfer & Wheeler)
def michalewicz(x, m=10):
    if not list_or_ndarray(x):
        return np.inf
    summation = 0
    for i in range(len(x)):
        summation += np.sin(x[i]) * (np.sin(((i+1)/np.pi) * (x[i] ** 2)) ** (2 * m))
    return -summation


# 2 dimensions, long curved valley, global minimum of 0 at (a,a^2)
def rosenbrock(x=None, y=None, z=None, a=1, b=5):
    if x is not None and list_or_ndarray(x) and np.array(x).size == 2:
        return (a - x[0]) ** 2 + b * (x[1] - x[0] ** 2) ** 2
    elif (y is not None and not list_or_ndarray(y)) and (z is not None and not list_or_ndarray(z)):
        return (a - y) ** 2 + b * (z - y ** 2) ** 2
    else:
        return np.inf


# 2 dimensions, global minimum inside deep curved peak, most gradient descent move away
def wheeler(x=None, y=None, z=None, a=1.5):
    if x is not None and list_or_ndarray(x) and len(x) == 2:
        return -np.exp(-((x[0] * x[1] - a) ** 2) - (x[1] - a) ** 2)
    elif (y is not None and not list_or_ndarray(y)) and (z is not None and not list_or_ndarray(z)):
        return -np.exp(-((y * z - a) ** 2) - (z - a) ** 2)
    else:
        return np.inf


# 2D -> 2D, pareto frontier: r=1, (theta % 2pi) \in (0,pi/2) or r=-1, (theta % 2pi) \in (pi,3pi/2)
def circle(x=None, y=None, z=None):
    if x is not None and list_or_ndarray(x) and len(x) == 2:
        r = 0.5 + 0.5 * (2 * x[1] / (1 + x[1] ** 2))
        return np.array([1 - r * np.cos(x[0]), 1 - r * np.sin(x[0])])
    elif (y is not None and not list_or_ndarray(y)) and (z is not None and not list_or_ndarray(z)):
        r = 0.5 + 0.5 * (2 * z / (1 + z ** 2))
        return np.array([1 - r * np.cos(y), 1 - r * np.sin(y)])
    else:
        return np.array([np.inf, np.inf])
