import pandas as pd
import numpy as np

def load_dataset(path_to_csv):
    """
    function for reading data from csv
    and processing to return a 2D feature matrix and a vector of class
    :return:
    """
    # todo: implement
    data = pd.read_csv(path_to_csv)
    x = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    maxs = np.max(x, axis=0)
    mins = np.min(x, axis=0)
    x = (x - mins) / (maxs - mins)
    x = np.hstack((np.ones((x.shape[0], 1)), x))
    return x, y


def split_dataset(X, y, test_size=0.8, shuffle=True):
    """
    function for spliting dataset into train and test
    :param X:
    :param y:
    :param float test_size: the proportion of the dataset to include in the test split
    :param bool shuffle: whether to shuffle the data before splitting
    :return:
    """
    # todo: implement.
    dataset = np.c_[X, y]
    if shuffle:
        np.random.shuffle(dataset)
    X = dataset[:, :-1]
    y = dataset[:, -1]
    X_train, X_test = X[:int(X.shape[0] * test_size)], X[int(X.shape[0] * test_size):]
    y_train, y_test = y[:int(y.shape[0] * test_size)], y[int(y.shape[0] * test_size):]
    return X_train, y_train, X_test, y_test


def bagging_sampler(X, y):
    """
    Randomly sample with replacement
    Size of sample will be same as input data
    :param X:
    :param y:
    :return:
    """
    # todo: implement
    X_sample, y_sample = None, None
    assert X_sample.shape == X.shape
    assert y_sample.shape == y.shape
    return X_sample, y_sample
