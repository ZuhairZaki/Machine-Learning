import numpy as np

class LogisticRegression:
    def __init__(self,params):
        """
        figure out necessary params to take as input
        :param params:
        """
        # todo: implement
        self.learning_rate = params['learning_rate']
        self.no_iterations = params['no_iterations']

    def fit(self, X, y):
        """
        :param X:
        :param y:
        :return: self
        """
        assert X.shape[0] == y.shape[0]
        assert len(X.shape) == 2
        # todo: implement

    def predict(self, X):
        """
        function for predicting labels of for all datapoint in X
        :param X:
        :return:
        """
        # todo: implement
