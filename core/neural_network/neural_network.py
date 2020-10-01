import numpy as np
from core.activations.relu import __relu, __relu_prime
from core.activations.sigmoid import __sigmoid
from core.cost.logistic import __logistic_cost, calc_f1_score
from core.preprocessing.initialization import initialize_parameters


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


def update_params(parameters, cache, layers, m, epoch, iteration_in_epoch, iterations_per_batch):
    for i in range(len(layers), 0, -1):
        layer = layers[i-1]
        A_before = cache["A" + str(i-1)]
        dZ_current = cache["dZ" + str(i)]
        dW_current = np.dot(dZ_current, A_before.T) / m
        db_current = np.sum(dZ_current, axis=1, keepdims=True)/m
        w_str = "W" + str(i)
        b_str = "b" + str(i)
        learning_rate_function = layer["learning_rate"] if "learning_rate" in layer.keys(
        ) else lambda _: 0.01
        learning_rate = learning_rate_function(epoch)
        lambd = layer["lambd"] if "lambd" in layer.keys() else 0

        if "optimization" in layer.keys():
            optimization = layer["optimization"]
            if optimization["alg"] == 'momentum':
                beta_1 = optimization["beta_1"]

                v_dW_str = "v_dW" + str(i)
                v_db_str = "v_db" + str(i)

                # initialization
                if v_dW_str not in cache.keys():
                    print("Running ",
                          optimization["alg"], " optimization algorithm")
                    cache[v_dW_str] = 0
                if v_db_str not in cache.keys():
                    cache[v_db_str] = 0

                cache[v_dW_str], cache[v_db_str] = beta_1 * cache[v_dW_str] + \
                    (1 - beta_1) * dW_current, beta_1 * cache[v_db_str] + \
                    (1 - beta_1) * db_current

                bias_denominator = max(1 - np.power(
                    beta_1, epoch * iteration_in_epoch + iteration_in_epoch), 1)

                w_update_value = cache["v_dW" + str(i)] / bias_denominator
                b_update_value = cache["v_db" + str(i)] / bias_denominator
        else:
            w_update_value = dW_current
            b_update_value = db_current

        parameters[w_str] = parameters[w_str] - learning_rate * \
            (w_update_value + (lambd/m)*parameters[w_str])
        parameters[b_str] = parameters[b_str] - learning_rate * b_update_value
    return parameters


def train(X, Y, X_dev, Y_dev, epochs=1000, batch_size=64, layers=[{"units": 1, "activation": 'sigmoid', "keep_prob": 1, "lambd": 0}]):
    print("X.shape", X.shape)
    print("Y.shape", Y.shape)

    m = Y.shape[0]
    cache = {}

    layer_dimensions = []
    for i in range(0, len(layers)):
        layer_dimensions.append(layers[i]["units"])

    layer_dimensions.insert(0, X.shape[0])
    parameters = initialize_network_parameters(layer_dimensions, m)

    error = 0
    acc = 0
    count = 1

    final_lambd = layers[len(
        layers)-1]["lambd"] if "lambd" in layers[len(layers)-1].keys() else 0

    iterations_per_batch = np.maximum(int(np.ceil(m/batch_size))-1, 1)

    for i in range(0, epochs):
        for j in range(0, iterations_per_batch):
            X_batch = X.T[j*batch_size: (j+1)*batch_size].T
            Y_batch = Y.T[j*batch_size: (j+1)*batch_size].T

            cache["A0"] = X_batch
            cache, parameters = forward_step(parameters, cache, layers)
            cache, parameters = backward_step(
                parameters, cache, layers, Y_batch)
            parameters = update_params(
                parameters, cache, layers, m, i, j, iterations_per_batch)

            AL = cache["A" + str(len(layers))]

            error += __logistic_cost(AL, Y_batch,
                                     parameters, final_lambd, len(layers))
            acc += calc_f1_score(AL, Y_batch)
            count += 1

            if(j + 1 == int(np.ceil(m/batch_size))-1):
                error /= count
                acc /= count
                count = 1
                if(i % 10 == 0):
                    dev_predictions = predict(X_dev, parameters, layers)
                    dev_error = __logistic_cost(
                        dev_predictions, Y_dev, parameters, 0, len(layers))
                    dev_acc = calc_f1_score(dev_predictions, Y_dev)
                    print('Error at epoch', i, '/', epochs, ': ',  error, "F1 Accuracy: ",
                          acc, '%', ' Dev error:', dev_error, ' Dev F1 Accuracy:, ', dev_acc, '%')

    predictions = predict(X, parameters, layers)
    error = __logistic_cost(predictions, Y, parameters,
                            final_lambd, len(layers))
    acc = calc_f1_score(predictions, Y)
    return parameters, predictions, error, acc


def predict(X_input, parameters, layers):
    cache = {}
    cache["A0"] = X_input
    cache, _ = forward_step(parameters, cache, layers, apply_dropout=False)
    return cache["A" + str(len(layers))]
