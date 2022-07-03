from optimization import optimize
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.integrate import odeint
import matplotlib.pyplot as plt


def sir_grad(y, t, u, _):
    infection, recovery, death = 0.2, 0.02, 0.01
    control = u(t)
    S, I, R = y
    dS = -infection * (1 - control) * S * I
    dI = infection * (1 - control) * S * I - recovery * I - death * I
    dR = recovery * I
    return [dS, dI, dR]


def cost_calc(S, I, u, t):
    infection, death = 0.2, 0.01
    death_cost = 1
    infection_cost = 1
    control_cost = 0.01
    control = u(t)
    cost = np.sum(death_cost * death * I +
                  infection_cost * infection * (1 - control) * S * I +
                  control_cost * control * S)
    return cost


def cost_eval(u):
    cubic_spline = CubicSpline(np.linspace(0, 100, len(u)), u)
    t = np.linspace(0, 100, 1001)
    y0 = [1, 0.02, 0]
    solution = odeint(sir_grad, y0, t, args=(cubic_spline, None))
    return cost_calc(solution[:, 0], solution[:, 1], cubic_spline, t)


if __name__ == "__main__":
    h = optimize(cost_eval, None, opt_method="divided_rectangles", interval_low=np.zeros(20), interval_high=np.ones(20),
                 value_tol=0.0001, size_tol=0.0001, max_steps=150)
    print(h)
    print(cost_eval(h))
    cubic_spline = CubicSpline(np.linspace(0, 100, len(h)), h)
    t = np.linspace(0, 100, 1001)
    y0 = [1, 0.02, 0]
    solution = odeint(sir_grad, y0, t, args=(cubic_spline, None))
    plt.plot(t, solution[:, 0], label="susceptible")
    plt.plot(t, solution[:, 1], label="infected")
    plt.plot(t, solution[:, 2], label="recovered")
    plt.plot(t, cubic_spline(t), label="control percentage")
    plt.legend()
    plt.show()
