import numpy as np


def split_dataset(X, y, shuffle, train_percentage=0.80, dev_percentage=0.1):
    #  add assertion that X and y have same length
    #  TODO: add shuffle
    m = y.shape[0]
    train_start = 0
    train_end = int(np.floor(train_percentage * m))
    dev_start = int(train_end + 1)
    dev_end = int(dev_start + np.floor(dev_percentage * m))
    test_start = int(dev_end + 1)
    test_end = m - 1

    return X[train_start:train_end], X[dev_start:dev_end], X[test_start:test_end], \
        y[train_start:train_end], y[dev_start:dev_end], y[test_start:test_end],
