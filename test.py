import numpy as np
import tensorflow as tf
from sklearn import datasets

from core.neural_network import neural_network
from core.simple.logistic_regression import regression
from core.simple.neural_network_2_layer import nn_2_layer
from core.preprocessing.split_dataset import split_dataset
from core.cost.logistic import calc_precision, __logistic_cost

X, y = datasets.load_breast_cancer(return_X_y=True)


def nnlib_regression(X, y):
    parameters, predictions, error, acc = regression(X.T, y, 10000)
    print(1*(predictions > 0.5))
    print("Accuracy = ", acc, '%')


def nnlib_nn_2_layer(X, y):
    parameters, predictions, error, acc = nn_2_layer(X.T, y, 10000, 16, 0.001)
    print(1*(predictions > 0.5))
    print("Accuracy = ", acc, '%')


def nnlib_nn_l_layer(X, y):
    predictions, error, acc = neural_network.train(
        X.T, y, 100000, [3, 4, 3, 1], 0.01)
    print(1*(predictions > 0.5))
    print("Accuracy = ", acc, '%')

# nnlib_regression(X, y)
# nnlib_nn_2_layer(X, y)


X_train, X_dev, X_test, y_train, y_dev, y_test = split_dataset(X, y, 0.6, 0.2)
network_layers = [8, 6, 4, 3, 1]
parameters, predictions, error, acc = neural_network.train(
    X.T, y, 100000, network_layers, 0.05)

predictions = neural_network.predict(
    X_dev.T, parameters, len(network_layers) - 1)
dev_error = __logistic_cost(predictions, y_dev, y_dev[0])
dev_acc = calc_precision(predictions, y_dev)

print("Train:\n\tError=", error, "\n\tAccuracy=", acc, '%')
print("Dev:\n\tError=", dev_error, "\n\tAccuracy=", dev_acc, '%')
