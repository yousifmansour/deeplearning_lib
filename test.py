from sklearn import datasets
import numpy as np
import tensorflow as tf

import nnlib
from nnlib import nn_2_layer, regression, nn_l_layer

X, y = datasets.load_breast_cancer(return_X_y=True)


def nnlib_regression():
    predictions, error, acc = regression(X.T, y, 10000)
    print(1*(predictions > 0.5))
    print("Accuracy = ", acc, '%')


def nnlib_nn_2_layer():
    predictions, error, acc = nn_2_layer(X.T, y, 10000, 64, 0.001)
    print(1*(predictions > 0.5))
    print("Accuracy = ", acc, '%')


def nnlib_nn_l_layer():
    predictions, error, acc = nn_l_layer(X.T, y, 10000, 8, 4, 0.001)
    print(1*(predictions > 0.5))
    print("Accuracy = ", acc, '%')


# nnlib_regression()
# nnlib_nn_2_layer()
# nnlib_nn_l_layer()
