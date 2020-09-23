import numpy as np
import tensorflow as tf
from sklearn import datasets

from core.logistic_regression import regression
from core.neural_network_2_layer import nn_2_layer
from core.neural_network.neural_network import nn_l_layer

X, y = datasets.load_breast_cancer(return_X_y=True)


def nnlib_regression():
    predictions, error, acc = regression(X.T, y, 10000)
    print(1*(predictions > 0.5))
    print("Accuracy = ", acc, '%')


def nnlib_nn_2_layer():
    predictions, error, acc = nn_2_layer(X.T, y, 10000, 16, 0.001)
    print(1*(predictions > 0.5))
    print("Accuracy = ", acc, '%')


def nnlib_nn_l_layer():
    predictions, error, acc = nn_l_layer(X.T, y, 10000, [4, 64, 8, 1], 0.001)
    print(1*(predictions > 0.5))
    print("Accuracy = ", acc, '%')


# nnlib_regression()
# nnlib_nn_2_layer()
nnlib_nn_l_layer()
