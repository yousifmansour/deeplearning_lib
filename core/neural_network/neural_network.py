import numpy as np
from sklearn.metrics import log_loss

from core.preprocessing.initialization import initialize_parameters
from core.cost.logistic import __logistic_cost, calc_precision
from core.activations.sigmoid import __sigmoid
from core.activations.relu import __relu, __relu_prime


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