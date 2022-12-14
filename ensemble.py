from data_handler import bagging_sampler
import numpy as np
from linear_model import sigmoid

class BaggingClassifier:
    def __init__(self, base_estimator, n_estimator):
        """
        :param base_estimator:
        :param n_estimator:
        :return:
        """
        # todo: implement
        self.base_estimator = base_estimator
        self.n_estimator = n_estimator

    def fit(self, X, y):
        """
        :param X:
        :param y:
        :return: self
        """
        assert X.shape[0] == y.shape[0]
        assert len(X.shape) == 2
        # todo: implement
        # fit model
        self.estimators = []
        for i in range(self.n_estimator):
            X_sample, y_sample = bagging_sampler(X, y)
            self.estimators.append(self.base_estimator.fit(X_sample, y_sample))
        self.estimators = np.array(self.estimators)

    def predict(self, X):
        """
        function for predicting labels of for all datapoint in X
        apply majority voting
        :param X:
        :return:
        """
        # todo: implement
        # predict model
        predictions = np.array([np.round(sigmoid(X.dot(estimator))).astype(int) for estimator in self.estimators])
        m, n = predictions.shape
        y_pred = np.zeros(n)
        for i in range(n):
            y_pred[i] = np.argmax(np.bincount(predictions[:, i]))
        return y_pred

