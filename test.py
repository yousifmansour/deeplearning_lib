import numpy as np
import tensorflow as tf
from sklearn import datasets

from core.cost.logistic import __logistic_cost, calc_f1_score
from core.neural_network import neural_network
from core.preprocessing.normalize_inputs import apply_normalize, normalize
from core.preprocessing.split_dataset import split_dataset

X, y = datasets.load_breast_cancer(return_X_y=True)


m = X.shape[0]
y.reshape(m, 1)
learning_rate = 0.01
keep_prob = 1
lambd = 0
iterations = 1000

layers = [{"units": 16, "activation": 'relu', "keep_prob": 1, "lambd": lambd, "learning_rate": learning_rate},

          {"units": 16, "activation": 'relu', "keep_prob": 1,
              "lambd": lambd, "learning_rate": learning_rate},
          {"units": 16, "activation": 'relu', "keep_prob": 1,
              "lambd": lambd, "learning_rate": learning_rate},
          {"units": 16, "activation": 'relu', "keep_prob": 1,
              "lambd": lambd, "learning_rate": learning_rate},
          {"units": 16, "activation": 'relu', "keep_prob": 1,
              "lambd": lambd, "learning_rate": learning_rate},

          {"units": 16, "activation": 'relu', "keep_prob": 1,
              "lambd": lambd, "learning_rate": learning_rate},
          {"units": 16, "activation": 'relu', "keep_prob": 1,
              "lambd": lambd, "learning_rate": learning_rate},
          {"units": 16, "activation": 'relu', "keep_prob": 1,
              "lambd": lambd, "learning_rate": learning_rate},
          {"units": 16, "activation": 'relu', "keep_prob": 1,
              "lambd": lambd, "learning_rate": learning_rate},

          {"units": 16, "activation": 'relu', "keep_prob": 1,
              "lambd": lambd, "learning_rate": learning_rate},
          {"units": 16, "activation": 'relu', "keep_prob": 1,
              "lambd": lambd, "learning_rate": learning_rate},
          {"units": 16, "activation": 'relu', "keep_prob": 1,
              "lambd": lambd, "learning_rate": learning_rate},
          {"units": 16, "activation": 'relu', "keep_prob": 1,
              "lambd": lambd, "learning_rate": learning_rate},

          {"units": 16, "activation": 'relu', "keep_prob": 1,
              "lambd": lambd, "learning_rate": learning_rate},
          {"units": 16, "activation": 'relu', "keep_prob": 1,
              "lambd": lambd, "learning_rate": learning_rate},
          {"units": 16, "activation": 'relu', "keep_prob": 1,
              "lambd": lambd, "learning_rate": learning_rate},
          {"units": 16, "activation": 'relu', "keep_prob": 1,
              "lambd": lambd, "learning_rate": learning_rate},
          {"units": 1, "activation": 'sigmoid', "keep_prob": 1, "lambd": lambd, "learning_rate": learning_rate}]

X_train, X_dev, X_test, y_train, y_dev, y_test = split_dataset(
    X, y, True, 0.6, 0.20)
X_train, X_mean, X_standard_deviation = normalize(X_train)
X_dev = apply_normalize(X_dev, X_mean, X_standard_deviation)
X_test = apply_normalize(X_test, X_mean, X_standard_deviation)

parameters, predictions, error, acc = neural_network.train(
    X_train.T, y_train, iterations=iterations, batch_size=64, layers=layers)

dev_predictions = neural_network.predict(X_dev.T, parameters, layers)
dev_error = __logistic_cost(dev_predictions, y_dev, parameters, 0, len(layers))
dev_acc = calc_f1_score(dev_predictions, y_dev)

print("Train:\n\tError=", error, "\n\tF1 Accuracy=", acc, '%')
print("Dev:\n\tError=", dev_error, "\n\tF1 Accuracy=", dev_acc, '%')

# print(1*(predictions.T[10: 20].T > 0.5))
# print(y[10: 20])
