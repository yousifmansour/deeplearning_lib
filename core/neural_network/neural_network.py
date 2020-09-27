import numpy as np

from core.preprocessing.initialization import initialize_parameters
from core.cost.logistic import __logistic_cost, calc_precision
from core.activations.sigmoid import __sigmoid
from core.activations.relu import __relu, __relu_prime


def initialize_network_parameters(layer_dimensions, m):
    parameters = {}
    num_layers = len(layer_dimensions)
    for i in range(0, num_layers-1):
        input_hidden_units = layer_dimensions[i]
        output_hidden_units = layer_dimensions[i+1]
        _w, _b = initialize_parameters(
            input_hidden_units, output_hidden_units, m)
        parameters["W" + str(i+1)], parameters["b" + str(i+1)] = _w, _b
    return parameters


def forward_step(parameters, cache, layers, apply_dropout=True):
    for i in range(1, len(layers)+1):
        layer = layers[i-1]
        W_current = parameters["W" + str(i)]
        b_current = parameters["b" + str(i)]
        A_before = cache["A" + str(i - 1)]
        Z = np.dot(W_current, A_before) + b_current
        parameters["Z" + str(i)] = Z
        if layer["activation"] == 'sigmoid':
            cache["A" + str(i)] = __sigmoid(Z)
        else:
            cache["A" + str(i)] = __relu(Z)
        if apply_dropout:
            keep_prob = layer["keep_prob"] if "keep_prob" in layer.keys(
            ) else 1
            cache["A" + str(i)] = np.multiply(cache["A" + str(i)],
                                              1 * (np.random.rand(cache["A" + str(i)].shape[0], cache["A" + str(i)].shape[1]) < keep_prob))
            cache["A" + str(i)] /= keep_prob
    return cache, parameters


def backward_step(parameters, cache, layers, Y):
    for i in range(len(layers), 0, -1):
        if i == len(layers):
            # sigmoid
            AL = cache["A" + str(i)]
            cache["dZ" + str(i)] = AL - Y
        else:
            # relu
            W_next = parameters["W" + str(i+1)]
            dZ_next = cache["dZ" + str(i+1)]
            Z_current = parameters["Z" + str(i)]
            dA_current = np.dot(W_next.T, dZ_next)
            dZ_current = np.multiply(dA_current, __relu_prime(Z_current))
            cache["dZ" + str(i)] = dZ_current
    return cache, parameters


def update_params(parameters, cache, layers, m):
    for i in range(len(layers), 0, -1):
        layer = layers[i-1]
        A_before = cache["A" + str(i-1)]
        dZ_current = cache["dZ" + str(i)]
        dW_current = np.dot(dZ_current, A_before.T) / m
        db_current = np.sum(dZ_current, axis=1, keepdims=True)/m
        w_str = "W" + str(i)
        b_str = "b" + str(i)
        learning_rate = layer["learning_rate"] if "learning_rate" in layer.keys(
        ) else 0.01
        lambd = layer["lambd"] if "lambd" in layer.keys() else 0
        parameters[w_str] = parameters[w_str] - learning_rate * \
            (dW_current + (lambd/m)*parameters[w_str])
        parameters[b_str] = parameters[b_str] - learning_rate * db_current
    return parameters


def train(X, Y, iterations=1000, layers=[{"units": 1, "activation": 'sigmoid', "keep_prob": 1, "lambd": 0}]):
    print("X.shape", X.shape)
    print("Y.shape", Y.shape)

    m = Y.shape[0]
    cache = {}
    cache["A0"] = X

    layer_dimensions = []
    for i in range(0, len(layers)):
        layer_dimensions.append(layers[i]["units"])

    layer_dimensions.insert(0, X.shape[0])
    parameters = initialize_network_parameters(layer_dimensions, m)

    for i in range(0, iterations):
        cache, parameters = forward_step(parameters, cache, layers)
        cache, parameters = backward_step(parameters, cache, layers, Y)
        parameters = update_params(parameters, cache, layers, m)

        AL = cache["A" + str(len(layers))]
        if(i % 100 == 0):
            print('Error at step', i, '/', iterations, ': ',
                  __logistic_cost(AL, Y, parameters, layers[len(layers)-1]["lambd"] or 0, len(layers)), "Accuracy: ", calc_precision(AL, Y), '%')

    AL = cache["A" + str(len(layers))]
    return parameters, AL, __logistic_cost(AL, Y, parameters, layers[len(layers)-1]["lambd"] or 0, len(layers)), calc_precision(AL, Y)


def predict(X_input, parameters, layers):
    cache = {}
    cache["A0"] = X_input
    cache, _ = forward_step(parameters, cache, layers, apply_dropout=False)
    return cache["A" + str(len(layers))]
