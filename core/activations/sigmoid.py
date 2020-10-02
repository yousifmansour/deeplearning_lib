import numpy as np


def __sigmoid(x):
    return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))


def __sigmoid_prime(x):
    return np.multiply(__sigmoid(x), 1-__sigmoid(x))
