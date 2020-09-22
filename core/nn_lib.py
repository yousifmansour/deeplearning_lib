import numpy as np
from sklearn.metrics import log_loss

from core.nn_helpers import __logistic_cost, __relu, __relu_prime, __sigmoid, __sigmoid_prime
from core.nn_helpers import initialize_parameters, calc_precision


def __initialize_nn_parameters(input_length, num_hidden_units, output_length, num_layers, m):
    parameters = {}
    if num_layers > 1:
        _w, _b = initialize_parameters(input_length, num_hidden_units, m)
        parameters["W1"], parameters["b1"] = _w, _b
        for i in range(2, num_layers):
            _w, _b = initialize_parameters(
                num_hidden_units, num_hidden_units, m)
            parameters["W" + str(i)], parameters["b" + str(i)] = _w, _b
        _w, _b = initialize_parameters(num_hidden_units, output_length, m)
        parameters["W" + str(num_layers)] = _w
        parameters["b" + str(num_layers)] = _b
    else:
        _w, _b = initialize_parameters(input_length, output_length, m)
        parameters["W1"], parameters["b1"] = _w, _b
    return parameters


def __nn_l_layer_forward(parameters, cache, num_layers):
    for layer in range(1, num_layers+1):
        W_current = parameters["W" + str(layer)]
        b_current = parameters["b" + str(layer)]
        A_before = cache["A" + str(layer - 1)]
        Z = np.dot(W_current, A_before) + b_current
        parameters["Z" + str(layer)] = Z
        if layer == num_layers + 1:
            cache["A" + str(layer)] = __sigmoid(Z)
        else:
            cache["A" + str(layer)] = __relu(Z)
    return cache, parameters


def __nn_l_layer_backward(parameters, cache, num_layers, Y):
    for layer in range(num_layers, 0, -1):
        if layer == num_layers:
            # sigmoid
            AL = cache["A" + str(layer)]
            cache["dZ" + str(layer)] = AL - Y
        else:
            # relu
            W_next = parameters["W" + str(layer+1)]
            dZ_next = cache["dZ" + str(layer+1)]
            Z_current = parameters["Z" + str(layer)]
            dA_current = np.dot(W_next.T, dZ_next)
            dZ_current = np.multiply(dA_current, __relu_prime(Z_current))
            cache["dZ" + str(layer)] = dZ_current
    return cache, parameters


def __nn_l_layer_update_params(parameters, cache, num_layers, learning_rate, m):
    for layer in range(num_layers, 0, -1):
        A_before = cache["A" + str(layer-1)]
        dZ_current = cache["dZ" + str(layer)]
        dW_current = np.dot(dZ_current, A_before.T) / m
        db_current = np.sum(dZ_current, axis=1, keepdims=True)/m
        w_str = "W" + str(layer)
        b_str = "b" + str(layer)
        parameters[w_str] = parameters[w_str] - learning_rate * dW_current
        parameters[b_str] = parameters[b_str] - learning_rate * db_current
    return parameters


def nn_l_layer(X, Y, iterations=1000, l=2, num_hidden_units=10, learning_rate=0.001):
    print("X.shape", X.shape)
    print("Y.shape", Y.shape)

    m = Y.shape[0]
    cache = {}
    cache["A0"] = X

    # 1) init params
    parameters = __initialize_nn_parameters(
        X.shape[0], num_hidden_units, 1, l, m)

    for i in range(0, iterations):
        cache, parameters = __nn_l_layer_forward(parameters, cache, l)
        cache, parameters = __nn_l_layer_backward(parameters, cache, l, Y)
        parameters = __nn_l_layer_update_params(
            parameters, cache, l, learning_rate, m)

        AL = cache["A" + str(l)]
        if(i % 100 == 0):
            print('Error at step', i, '/', iterations, ': ',
                  __logistic_cost(AL, Y, m), "Accuracy: ", calc_precision(AL, Y), '%')

    AL = cache["A" + str(l)]
    return AL, __logistic_cost(AL, Y, m), calc_precision(AL, Y)


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

    return A2, __logistic_cost(A2, Y, m), calc_precision(A2, Y)


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
