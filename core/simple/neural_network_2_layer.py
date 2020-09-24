import numpy as np

from core.activations.relu import __relu, __relu_prime
from core.activations.sigmoid import __sigmoid
from core.cost.logistic import __logistic_cost, calc_precision
from core.preprocessing.initialization import initialize_parameters


def nn_2_layer(X, Y, iterations=1, num_hidden_units=1, learning_rate=0.001):
    # 1) init params
    m = Y.shape[0]
    W1, b1 = initialize_parameters(X.shape[0], num_hidden_units, m)
    W2, b2 = initialize_parameters(num_hidden_units, 1, m)

    for i in range(0, iterations):
        # 2) forward pass
        Z1 = np.dot(W1, X) + b1
        A1 = __relu(Z1)
        Z2 = np.dot(W2, A1) + b2
        A2 = __sigmoid(Z2)

        # 3) backward pass
        dZ2 = A2 - Y
        dW2 = np.dot(dZ2, A1.T) / m
        db2 = np.sum(dZ2, axis=1, keepdims=True)/m

        dA1 = np.dot(W2.T, dZ2)
        dZ1 = np.multiply(dA1, __relu_prime(Z1))
        dW1 = np.dot(dZ1, X.T)/m
        db1 = np.sum(dZ1, axis=1, keepdims=True)/m

        # 4) update params
        W1 = W1 - learning_rate * dW1
        b1 = b1 - learning_rate * db1
        W2 = W2 - learning_rate * dW2
        b2 = b2 - learning_rate * db2

        if(i % 100 == 0):
            print('Error at step', i, '/', iterations, ': ',
                  __logistic_cost(A2, Y, m), "Accuracy: ", calc_precision(A2, Y), '%')

    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2}, A2, __logistic_cost(A2, Y, m), calc_precision(A2, Y)
