import pandas as pd
import numpy as np
from ensemble import BaggingClassifier
from linear_model import LogisticRegression

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

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
    
load_dataset('data_banknote_authentication.csv')