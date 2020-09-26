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
iterations = 100000

layers = [{"units": 8, "activation": 'relu', "keep_prob": 1, "lambd": lambd, "learning_rate": learning_rate},
          {"units": 8, "activation": 'relu', "keep_prob": keep_prob,
              "lambd": lambd, "learning_rate": learning_rate},
          {"units": 4, "activation": 'relu', "keep_prob": keep_prob,
           "lambd": lambd, "learning_rate": learning_rate},
          {"units": 4, "activation": 'relu', "keep_prob": keep_prob,
           "lambd": lambd, "learning_rate": learning_rate},
          {"units": 4, "activation": 'relu', "keep_prob": keep_prob,
           "lambd": lambd, "learning_rate": learning_rate},
          {"units": 4, "activation": 'relu', "keep_prob": keep_prob,
           "lambd": lambd, "learning_rate": learning_rate},
          {"units": 1, "activation": 'sigmoid', "keep_prob": 1, "lambd": lambd, "learning_rate": learning_rate}]

X_train, X_dev, X_test, y_train, y_dev, y_test = split_dataset(X, y, 0.6, 0.20)
X_train, X_mean, X_standard_deviation = normalize(X_train)
X_dev = apply_normalize(X_dev, X_mean, X_standard_deviation)
X_test = apply_normalize(X_test, X_mean, X_standard_deviation)

parameters, predictions, error, acc = neural_network.train(
    X_train.T, y_train, iterations=iterations, layers=layers)

dev_predictions = neural_network.predict(X_dev.T, parameters, layers)
dev_error = __logistic_cost(dev_predictions, y_dev, y_dev.shape[0], parameters, layers[len(
    layers)-1]["lambd"], len(layers))


dev_acc = calc_precision(dev_predictions, y_dev)

print("Train:\n\tError=", error, "\n\tAccuracy=", acc, '%')
print("Dev:\n\tError=", dev_error, "\n\tAccuracy=", dev_acc, '%')
