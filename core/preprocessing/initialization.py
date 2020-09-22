import numpy as np


def initialize_parameters(n_input_layer, n_output_layer, m):
    W = np.random.randn(n_output_layer, n_input_layer) * 0.01
    b = np.zeros((n_output_layer, m))
    return W, b
