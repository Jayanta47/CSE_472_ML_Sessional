from data_handler import bagging_sampler
import pandas as pd

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
        self.estimators = []

    def fit(self, X, y):
        """
        :param X:
        :param y:
        :return: self
        """
        assert X.shape[0] == y.shape[0]
        assert len(X.shape) == 2
        # todo: implement
        for i in range(self.n_estimator):
            X_sample, y_sample = bagging_sampler(X, y)
            self.base_estimator.fit(X_sample, y_sample)
            self.estimators.append(self.base_estimator)

    def predict(self, X):
        """
        function for predicting labels of for all datapoint in X
        apply majority voting
        :param X:
        :return:
        """
        # todo: implement
        y_pred = []
        for estimator in self.estimators:
            y_pred.append(estimator.predict(X))
        ensembled_result =  pd.DataFrame(y_pred).mode(axis=0).values.flatten()
        return ensembled_result
