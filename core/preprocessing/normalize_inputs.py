import numpy as np


def normalize(X, Y):
    m = X.shape[0]

    X_mean = np.sum(X, axis=0) / m
    X_standard_deviation = np.sqrt(np.sum(np.multiply(X, X), axis=0) / m)
    X_normalized = np.divide((X - X_mean), X_standard_deviation)

    Y_mean = np.sum(Y, axis=0) / m
    Y_standard_deviation = np.sqrt(np.sum(np.multiply(Y, Y), axis=0) / m)
    Y_normalized = np.divide((Y - Y_mean), Y_standard_deviation)

    return X_normalized, Y_normalized, X_mean, X_standard_deviation, Y_mean, Y_standard_deviation


def apply_normalize(values, mean, standard_deviation):
    # return np.divide(values-mean, standard_deviation)
    # print(mean.shape)
    # print(values.shape)
    return values
