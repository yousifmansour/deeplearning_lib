
import numpy as np

from core.preprocessing.initialization import initialize_parameters
from core.activations.sigmoid import __sigmoid
from core.cost.logistic import __logistic_cost, calc_precision


def regression(X, Y, iterations=1, learning_rate=0.001):
    # 1) init params
    m = Y.shape[0]
    W, b = initialize_parameters(X.shape[0], 1, m)
    for i in range(0, iterations):
        # 2) forward pass
        Z = np.dot(W, X) + b
        A = __sigmoid(Z)
        # 3) backward pass calculating gradients
        dZ = A - Y
        dW = np.dot(dZ, X.T)
        db = dZ
        # 4) update params
        W = W - learning_rate * dW
        b = b - learning_rate * db
        if(i % 100 == 0):
            print('Error at step', i, ': ', __logistic_cost(A, Y, m),
                  "Accuracy: ", calc_precision(A, Y), '%')

    return A, __logistic_cost(A, Y, m), calc_precision(A, Y)
