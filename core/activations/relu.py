import numpy as np


def __relu(x):
    return np.multiply(x, (x > 0))


def __relu_prime(x):
    return 1 * (x > 0)
