import numpy as np
import tensorflow as tf
from sklearn import datasets

from core.neural_network import neural_network
from core.simple.logistic_regression import regression
from core.simple.neural_network_2_layer import nn_2_layer
from core.preprocessing.split_dataset import split_dataset
from core.cost.logistic import calc_precision, __logistic_cost
from core.preprocessing.normalize_inputs import normalize, apply_normalize

X, y = datasets.load_breast_cancer(return_X_y=True)


def nnlib_regression(X, y):
    parameters, predictions, error, acc = regression(X.T, y, 10000)
    print(1*(predictions > 0.5))
    print("Accuracy = ", acc, '%')


def nnlib_nn_2_layer(X, y):
    parameters, predictions, error, acc = nn_2_layer(X.T, y, 10000, 16, 0.001)
    print(1*(predictions > 0.5))
    print("Accuracy = ", acc, '%')


# nnlib_regression(X, y)
# nnlib_nn_2_layer(X, y)

learning_rate = 0.001
keep_prob = 1
lambd = 0

layers = [{"units": 4, "activation": 'relu', "keep_prob": 1, "lambd": lambd, "learning_rate": learning_rate},
          {"units": 4, "activation": 'relu', "keep_prob": keep_prob,
              "lambd": lambd, "learning_rate": learning_rate},
          {"units": 4, "activation": 'relu', "keep_prob": keep_prob,
              "lambd": lambd, "learning_rate": learning_rate},
          {"units": 4, "activation": 'relu', "keep_prob": keep_prob,
           "lambd": lambd, "learning_rate": learning_rate},
          {"units": 1, "activation": 'sigmoid', "keep_prob": 1, "lambd": lambd, "learning_rate": learning_rate}]

X_train, X_dev, X_test, y_train, y_dev, y_test = split_dataset(X, y, 0.7, 0.15)
X_train, y_train, X_mean, X_standard_deviation, y_mean, y_standard_deviation = normalize(
    X_train, y_train)
X_dev = apply_normalize(X_dev, X_mean, X_standard_deviation)
print(X_train.shape)
print(X_dev.shape)
# X_test = apply_normalize(X_test, X_mean, X_standard_deviation)
# y_dev = apply_normalize(y_dev, y_mean, y_standard_deviation)
# y_test = apply_normalize(y_test, y_mean, y_standard_deviation)

# parameters, predictions, error, acc = neural_network.train(X.T, y, iterations=1000, layers=layers)


# predictions = neural_network.predict(X_dev.T, parameters, layers)
# dev_error = __logistic_cost(
#     predictions, y_dev, y_dev[0], layers[len(layers)-1]["lambd"], parameters, len(layers))
# dev_acc = calc_precision(predictions, y_dev)

# print("Train:\n\tError=", error, "\n\tAccuracy=", acc, '%')
# print("Dev:\n\tError=", dev_error, "\n\tAccuracy=", dev_acc, '%')
