import numpy as np
from sklearn.metrics import log_loss


def __logistic_cost(A, Y, m):
    return log_loss(Y.reshape(A.shape).T, A.T)


def calc_precision(y_hat, y):
    # Precision = TruePositives / (TruePositives + FalsePositives)
    true_positive = np.sum(np.multiply(1*(y_hat > 0.5), y))
    false_positive = np.sum(np.multiply(1*(y_hat > 0.5), 1*(y == 0)))
    return 100 * true_positive / (1+true_positive + false_positive)
