import numpy as np


def initialize_parameters(n_input_layer, n_output_layer, m):
    W = np.random.randn(n_output_layer, n_input_layer) * \
        np.sqrt(2/n_output_layer)
    b = np.zeros((n_output_layer, 1))
    return W, b
