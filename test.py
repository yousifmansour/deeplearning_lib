import numpy as np
from sklearn import datasets
from sklearn.linear_model import LinearRegression
from tensorflow import keras
import tensorflow as tf

import nnlib
from nnlib import nn_2_layer, regression, nn_l_layer

X, y = datasets.load_breast_cancer(return_X_y=True)
# X = np.asarray([[2, 3], [4, 6], [-1, -32], [123, 1234], [12, 11]])
# y = np.asarray([[1], [1], [0], [1], [0]]).reshape(1, 5)


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
nnlib_nn_2_layer()
# nnlib_nn_l_layer()
