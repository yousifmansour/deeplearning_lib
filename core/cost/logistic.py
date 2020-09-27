import numpy as np
from sklearn.metrics import log_loss


def __logistic_cost(predictions, Y, parameters, lambd=0, num_layers=0):
    m = Y.shape[0]
    regularization_cost = 0
    for i in range(0, num_layers):
        regularization_cost += np.linalg.norm(
            parameters["W" + str(i + 1)], 'fro')
    return log_loss(Y.reshape(predictions.shape).T, predictions.T) + (lambd * regularization_cost/(2*m))


def calc_precision(y_hat, y):
    # Precision = TruePositives / (TruePositives + FalsePositives)
    true_positive = np.sum(np.multiply(1*(y_hat > 0.5), y))
    false_positive = np.sum(np.multiply(1*(y_hat > 0.5), 1*(y == 0)))
    return 100 * true_positive / (1+true_positive + false_positive)
