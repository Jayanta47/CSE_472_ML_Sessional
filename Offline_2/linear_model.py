import numpy as np


class LogisticRegression:
    def __init__(self, params):
        """
        figure out necessary params to take as input
        :param params:
        """
        # todo: implement
        self.lr = params['learning_rate']
        self.num_iter = params['num_iter']
        self.verbose = params['verbose']

    def __sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def __loss(self, y_hat, y):
        return (-y * np.log(y_hat) - (1 - y) * np.log(1 - y_hat)).mean()

    def fit(self, X, y):
        """
        :param X:
        :param y:
        :return: self
        """
        assert X.shape[0] == y.shape[0]
        assert len(X.shape) == 2
        # todo: implement

        # weights initialization
        self.theta = np.zeros(X.shape[1])
        self.b = 0

        # Normalize X
        # X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

        for i in range(self.num_iter):
            z = np.dot(X, self.theta)
            y_hat = self.__sigmoid(z + self.b)

            d_theta = np.dot(X.T, (y_hat - y)) / X.shape[0]
            self.theta -= self.lr * d_theta

            d_b = np.sum(y_hat - y) / X.shape[0]
            self.b -= self.lr * d_b

            # z = np.dot(X, self.theta)
            # y_hat = self.__sigmoid(z)
            loss = self.__loss(y_hat, y)

            if (self.verbose == True and i % 1000 == 0):
                print(f'loss: {loss} \t')

    def predict(self, X):
        """
        function for predicting labels of for all datapoint in X
        :param X:
        :return:
        """
        # todo: implement

        # Normalize X
        return self.__sigmoid(np.dot(X, self.theta)+self.b) >= 0.5
