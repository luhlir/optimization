import numpy as np


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