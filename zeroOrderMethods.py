from lineSearch import line_search
import numpy as np
import multiprocessing


def multiprocessing_eval(f, x, f_args):
    return f(x, **f_args)


def batch_eval(f, X, f_args, multithreaded):
    n = len(X)
    if multithreaded:
        pool = multiprocessing.Pool(multiprocessing.cpu_count() - 1)
        args = []
        for x in X:
            args.append((f, x, f_args))
        y = pool.starmap(multiprocessing_eval, args)
        pool.close()
    else:
        y = np.zeros(n)
        for i in range(n):
            y[i] = f(X[i], **f_args)
    return y


def coordinate_descent(f, x_0, f_args, accel=False, lin_method="full_search", lin_args=None, max_steps=50, tol=0.00001):
    if lin_args is None:
        lin_args = {}
    x_curr = np.array(x_0)
    x_prev = np.ones(len(x_0)) * np.inf
    while max_steps > 0 and np.linalg.norm(x_curr - x_prev) > tol:
        x_prev = x_curr.copy()
        for i in range(len(x_0)):
            d = np.zeros(len(x_0))
            d[i] = 1
            x_curr = line_search(f, x_curr, f_args, d, lin_method, lin_args)
        if accel:
            diag = x_curr - np.array(x_0)
            x_curr = line_search(f, x_curr, f_args, diag / np.linalg.norm(diag), lin_method, lin_args)
        max_steps -= 1
    return x_curr


def powells_method(f, x_0, f_args, lin_method='full_search', lin_args=None, max_steps=50, tol=0.00001):
    n = len(x_0)
    U = np.identity(n)
    last = np.zeros(n)
    last[n-1] = 1
    reset = False

    x_curr = np.array(x_0)
    x_prev = np.ones(n) * np.inf
    while max_steps > 0 and np.linalg.norm(x_curr - x_prev) > tol:
        x_prev = x_curr.copy()

        for i in range(n):
            x_curr = line_search(f, x_curr, f_args, U[i], lin_method, lin_args)
            U[i] = U[(i+1) % n].copy()
        U[n-1] = x_prev - x_curr
        x_curr = line_search(f, x_curr, f_args, U[n-1], lin_method, lin_args)
        max_steps -= 1

        if reset:
            U = np.identity(n)
            reset = False
        elif all(U[0][i] == last[i] for i in range(n)):
            reset = True
    return x_curr


def hooke_jeeves_method(f, x_0, f_args, alpha=1, beta=0.5, max_steps=50, tol=0.001, multithreaded=False):
    n = len(x_0)
    D = []
    for i in range(n):
        a = np.zeros(n)
        a[i] = 1
        D.append(a.copy())
        D.append(-a)

    return pattern_search(f, x_0, f_args, D, alpha, beta, max_steps, tol, False, multithreaded)


def make_minimal_positive_spanning_set(a, n):
    L = np.zeros((n, n))
    sigma = round(1 / np.sqrt(a))

    for i in range(n):
        L[i][i] = sigma * (2 * np.random.random() - 1)
        if not a == 1:
            for j in range(i + 1, n):
                L[i][j] = (2 * np.random.random() * (sigma - 1)) - sigma + 1
    D = L[:, np.random.permutation(range(n))]
    D = D[np.random.permutation(range(n)), :]
    D = np.concatenate((D, -np.sum(D, 0, keepdims=True)), 0)
    return D


def pattern_search(f, x_0, f_args, D=None, alpha=1, beta=0.5, max_steps=50, tol=0.001,
                   opportunistic=True, multithreaded=False):
    n = len(x_0)
    x_min = x_0.copy()

    if D is None:
        D = make_minimal_positive_spanning_set(1, n)

    f_min = f(x_min, **f_args)
    while max_steps > 0 and alpha > tol:
        if opportunistic:
            for i in range(len(D)):
                x = x_min + alpha * D[i]
                y = f(x, **f_args)
                if y < f_min:
                    x_min = x.copy()
                    f_min = y
                    # Move this direction to the top of the directions
                    d = D[i].copy()
                    for j in range(i):
                        d_temp = D[j].copy()
                        D[j] = d.copy()
                        d = d_temp.copy()
                    D[i] = d.copy()
                    print(x_min)
                    break
            else:
                alpha *= beta
        else:
            X = D.copy()
            for i in range(len(D)):
                X[i] = x_min + alpha * D[i]
            Y = batch_eval(f, X, f_args, multithreaded)
            if np.min(Y) < f_min:
                x_min = X[np.argmin(Y)].copy()
                f_min = np.min(Y)
            else:
                alpha *= beta
        max_steps -= 1
    return x_min


def nelder_mead_simplex(f, f_args, S=None, x_0=None, side_0=1, reflect=1, expansion=2, contraction=0.5, shrinkage=0.5,
                        max_steps=50, tol=0.001, multithreaded=False):
    if S is None:
        if x_0 is None:
            return np.inf
        # Algorithmically determine a regular simplex centered around a point
        n = len(x_0)
        S = np.zeros((n + 1, n))
        one = np.ones(n)
        scale = np.array(x_0) - (side_0 / (n * np.sqrt(2))) * (1 + (1 / np.sqrt(n + 1))) * one
        for i in range(n):
            basis = np.zeros(n)
            basis[i] = side_0 / np.sqrt(2)
            S[i] = basis + scale
        S[n] = (side_0 / np.sqrt(2 * (n + 1))) * one + np.array(x_0)
    else:
        n = len(S[0])

    # Let the simplex crawl across the space
    Y = batch_eval(f, S, f_args, multithreaded)
    while max_steps > 0 and np.std(Y) > tol:
        indices = np.argsort(Y)
        S, Y = S[indices], Y[indices]
        x_avg = np.mean(S[:-1], 0)
        xr = x_avg + reflect * (x_avg - S[-1])
        yr = f(xr, **f_args)
        if yr < Y[0]:   # Go even further
            xe = x_avg + expansion * (xr - x_avg)
            ye = f(xe, **f_args)
            if ye < yr: # Was expansion worth it?
                S[-1], Y[-1] = xe.copy(), ye
            else:
                S[-1], Y[-1] = xr.copy(), yr
        elif yr < Y[-2]:    # Contraction isn't going to help
            S[-1], Y[-1] = xr.copy(), yr
        else:   # Let's do a contraction
            if yr < Y[-1]:
                S[-1], Y[-1] = xr.copy(), yr
            xc = x_avg + contraction * (S[-1] - x_avg)
            yc = f(xc, **f_args)
            if yc > Y[-1]:  # We will do a shrinkage instead
                for i in range(n + 1):
                    S[i] = (S[i] + S[0]) * shrinkage
                Y = batch_eval(f, S, f_args, multithreaded)
            else:
                S[-1], Y[-1] = xc.copy(), yc
        max_steps -= 1
    return S[np.argmin(Y)]


def f_unit(f, x, f_args, interval_low, interval_high):
    return f(x * (interval_high - interval_low) + interval_low, **f_args)


class Interval:
    def __init__(self, depths, center, y):
        self.depths = depths.copy()
        self.center = center.copy()
        self.y = y
        self.key = str(self.vertex_dist())

    def __lt__(self, other):
        return self.y < other.y

    def __gt__(self, other):
        return self.y > other.y

    def __le__(self, other):
        return self.y <= other.y

    def __ge__(self, other):
        return self.y >= other.y

    def __eq__(self, other):
        return self.y == other.y

    def vertex_dist(self):
        return np.linalg.norm(0.5 * (3 ** (-self.depths)))

    @staticmethod
    def divide(interval, f, f_args, interval_low, interval_high):
        depth = np.min(interval.depths)
        movement = 3 ** (-depth - 1)
        direction = np.argmin(interval.depths)
        new_depths = interval.depths.copy()
        new_depths[direction] += 1
        c0 = interval.center.copy()
        c2 = interval.center.copy()
        c0[direction] += movement
        c2[direction] -= movement
        int0 = Interval(new_depths, c0, f_unit(f, c0, f_args, interval_low, interval_high))
        int1 = Interval(new_depths, interval.center, interval.y)
        int2 = Interval(new_depths, c2, f_unit(f, c2, f_args, interval_low, interval_high))
        return [int0, int1, int2]


def divided_rectangles(f, f_args, interval_low, interval_high,
                       max_steps=50, size_tol=0.001, value_tol=0.001, multithreaded=False):

    def f_unit_inv(x):
        return x * (interval_high - interval_low) + interval_low

    if multithreaded:
        pool = multiprocessing.Pool(multiprocessing.cpu_count() - 1)

    n = len(interval_low)
    c = np.ones(n) * 0.5
    interval0 = Interval(np.zeros(n), c, f_unit(f, c, f_args, interval_low, interval_high))
    widths = {interval0.key: [interval0]}
    best = interval0

    # Start the whole thing off!
    while max_steps > 0 and best.vertex_dist() > size_tol:
        # Get the optimal intervals
        optimal = []
        empty = 0
        for key in widths:
            if len(widths[key]) > 0:
                optimal.append(np.sort(widths[key])[0])
            else:
                empty += 1
        optimal = sorted(optimal, key=lambda interval: interval.vertex_dist())
        if len(optimal) > 2:
            x, y = optimal[0].vertex_dist(), optimal[0].y
            x1, y1 = optimal[1].vertex_dist(), optimal[1].y
            i = 2
            while i < len(optimal):
                # If the previous optimal sample is above the line between its neighbors, remove it
                x2, y2 = optimal[i].vertex_dist(), optimal[i].y
                slope = (y2 - y) / (x2 - x)
                if y1 > slope * (x1 - x) + y + value_tol:
                    optimal.remove(optimal[i-1])
                    x1, y1 = x2, y2
                else:
                    i += 1
                    x, y = x1, y1
                    x1, y1 = x2, y2

        # Divide all of the potentially optimal points!
        if multithreaded:
            args = []
            for interval in optimal:
                widths[interval.key].remove(interval)
                args.append((interval, f, f_args, interval_low, interval_high))
            pool_returned = pool.starmap(Interval.divide, args)
            for new_ints in pool_returned:
                for new_int in new_ints:
                    if new_int.key not in widths.keys():
                        widths[new_int.key] = [new_int]
                    else:
                        widths[new_int.key].append(new_int)
                best = min(min(new_ints), best)
        else:
            for interval in optimal:
                widths[interval.key].remove(interval)
                new_ints = Interval.divide(interval, f, f_args, interval_low, interval_high)
                for new_int in new_ints:
                    if new_int.key not in widths.keys():
                        widths[new_int.key] = [new_int]
                    else:
                        widths[new_int.key].append(new_int)
                best = min(min(new_ints), best)
        max_steps -= 1

    if multithreaded:
        pool.close()
    return f_unit_inv(best.center)
