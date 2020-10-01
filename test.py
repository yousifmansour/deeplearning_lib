import numpy as np
import tensorflow as tf
from sklearn import datasets

from core.cost.logistic import __logistic_cost, calc_f1_score
from core.neural_network import neural_network
from core.preprocessing.normalize_inputs import apply_normalize, normalize
from core.preprocessing.split_dataset import split_dataset

X, y = datasets.load_breast_cancer(return_X_y=True)
np.random.seed(1)


m = X.shape[0]
y.reshape(m, 1)


epochs = 100


def learning_rate(epoch):
    return (0.05 if epoch < epochs else 0.01)


keep_prob = 0.6
lambd = 0

hidden_units = 1024

optimization_alg = "momentum"
beta_1 = 0.9

hidden_layer = {"units": hidden_units, "activation": 'relu', "keep_prob": keep_prob, "lambd": lambd,
                "learning_rate": learning_rate, "optimization": {"alg": optimization_alg, "beta_1": beta_1}}

layers = [
    {"units": hidden_units, "activation": 'relu', "keep_prob": 1,
     "lambd": lambd, "learning_rate": learning_rate,
     "optimization": {"alg": optimization_alg, "beta_1": beta_1}
     },

    hidden_layer,
    hidden_layer,
    hidden_layer,
    hidden_layer,

    {"units": 1, "activation": 'sigmoid', "keep_prob": 1, "lambd": lambd, "learning_rate": learning_rate,
     "optimization": {"alg": optimization_alg, "beta_1": beta_1}
     }, ]

X_train, X_dev, X_test, y_train, y_dev, y_test = split_dataset(
    X, y, True, 0.5, 0.25)
X_train, X_mean, X_standard_deviation = normalize(X_train)
X_dev = apply_normalize(X_dev, X_mean, X_standard_deviation)
X_test = apply_normalize(X_test, X_mean, X_standard_deviation)

parameters, predictions, error, acc = neural_network.train(
    X_train.T, y_train, X_dev.T, y_dev, epochs=epochs, batch_size=64, layers=layers)

dev_predictions = neural_network.predict(X_dev.T, parameters, layers)
dev_error = __logistic_cost(dev_predictions, y_dev, parameters, 0, len(layers))
dev_acc = calc_f1_score(dev_predictions, y_dev)

print("Train:\n\tError=", error, "\n\tF1 Accuracy=", acc, '%')
print("Dev:\n\tError=", dev_error, "\n\tF1 Accuracy=", dev_acc, '%')

# test_predictions = neural_network.predict(X_test.T, parameters, layers)
# test_error = __logistic_cost(
#     test_predictions, y_test, parameters, 0, len(layers))
# test_acc = calc_f1_score(test_predictions, y_test)
# print("Test:\n\tError=", test_error, "\n\tF1 Accuracy=", test_acc, '%')

# print(1*(predictions.T[10: 20].T > 0.5))
# print(y[10: 20])

# test_predictions = neural_network.predict(X_test.T, parameters, layers)s
