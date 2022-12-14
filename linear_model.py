import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class LogisticRegression:
    def __init__(self,learning_rate=0.1,epochs=1000):
        """
        figure out necessary params to take as input
        :param params:
        """
        # todo: implement
        self.learning_rate = learning_rate
        self.epochs = epochs

    def fit(self, X, y):
        """
        :param X:
        :param y:
        :return: self
        """
        assert X.shape[0] == y.shape[0]
        assert len(X.shape) == 2
        # todo: implement
        self.w = np.zeros(X.shape[1])
        for i in range(self.epochs):
            h = sigmoid(X.dot(self.w))
            self.w -= self.learning_rate * (X.T.dot(h - y) / len(y))

        return self.w

    def predict(self, X):
        """
        function for predicting labels of for all datapoint in X
        :param X:
        :return:
        """
        # todo: implement
        return np.round(sigmoid(X.dot(self.w))).astype(int)
