import numpy as np


def __sigmoid(x):
    return np.maximum(np.minimum(1 / (1 + np.exp(-x)), 0.999), 0.001)


def __sigmoid_prime(x):
    return np.multiply(__sigmoid(x), 1-__sigmoid(x))
