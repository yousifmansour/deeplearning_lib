import numpy as np
from sklearn.metrics import log_loss


def __logistic_cost(A, Y, m):
    return log_loss(Y.reshape(A.shape).T, A.T)


def __sigmoid(x):
    return np.maximum(np.minimum(1 / (1 + np.exp(-x)), 0.999), 0.001)


def __sigmoid_prime(x):
    return np.multiply(__sigmoid(x), 1-__sigmoid(x))


def __relu(x):
    return np.multiply(x, (x > 0))


def __relu_prime(x):
    return 1 * (x > 0)


def initialize_parameters(n_input_layer, n_output_layer, m):
    W = np.random.randn(n_output_layer, n_input_layer) * 0.01
    b = np.zeros((n_output_layer, m))
    return W, b


def calc_precision(y_hat, y):
    # Precision = TruePositives / (TruePositives + FalsePositives)
    true_positive = np.sum(np.multiply(1*(y_hat > 0.5), y))
    false_positive = np.sum(np.multiply(1*(y_hat > 0.5), 1*(y == 0)))
    return 100 * true_positive / (1+true_positive + false_positive)
