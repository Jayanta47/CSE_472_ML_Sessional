import numpy as np 
from data_handler import dimensionReducer

class GaussianMixtureModel:
    def __init__(self, n_components, visualizer=None, max_iter=1000, tol=1e-6 ):
        self.n_components = n_components
        self.max_iter = max_iter
        self.visualizer = visualizer
        self.tol = tol
        self.weights = None
        self.means = None
        self.covariances = None
        self.log_likelihoods = []

    def fit(self, X):
        n_features = X.shape[1]
        # Initialize the parameters randomly
        self.weights = np.random.rand(self.n_components)
        self.weights /= np.sum(self.weights)
        self.means = np.random.rand(self.n_components, n_features)
        self.covariances = np.array([np.eye(n_features) for _ in range(self.n_components)])


        for i in range(self.max_iter):
            if self.visualizer:
                if n_features > 2:
                    graph_X = dimensionReducer(X, dim=2)
                    graph_mean = dimensionReducer(self.means, dim=2)
                    graph_cov = np.zeros((self.n_components, 2, 2))
                    for j in range(self.n_components):
                        temp = dimensionReducer(self.covariances[j], dim=2)
                        temp = dimensionReducer(temp.T, dim=2).T
                        min_eig = np.min(np.real(np.linalg.eigvals(temp)))
                        if min_eig < 0:
                            temp -= 10*min_eig * np.eye(temp.shape[0])
                        graph_cov[j] = temp
                    # print("graph_mean: ", graph_mean.shape)
                    # print("graph_cov: ", graph_cov.shape)
                    # print("graph_X: ", graph_X.shape)
                else:
                    graph_X = X
                    graph_mean = self.means
                    graph_cov = self.covariances

                self.visualizer.visualize(graph_X, mean=graph_mean, cov=graph_cov,n_components=self.n_components)
            # E-step
            log_likelihood, responsibilities = self._e_step(X)
            
            self.log_likelihoods.append(log_likelihood)
            # M-step
            self._m_step(X, responsibilities)
            
            # check for convergence
            if (i > 0 and abs(self.log_likelihoods[-1] - self.log_likelihoods[-2]) < self.tol) or i==self.max_iter-1:
                print("iter: ", i, ": log: ",log_likelihood)
                break
        return self.log_likelihoods[-1]

    def _e_step(self, X):
        n_samples = X.shape[0]
        responsibilities = np.zeros((n_samples, self.n_components))
        log_likelihood = 0
        for k in range(self.n_components):
            responsibilities[:, k] = self.weights[k] * self.__pdf(X, self.means[k], self.covariances[k])
        log_likelihood = np.sum(np.log(np.sum(responsibilities, axis=1)))
        responsibilities /= np.sum(responsibilities, axis=1, keepdims=True)
        return log_likelihood, responsibilities

    def _m_step(self, X, responsibilities):
        n_samples = X.shape[0]
        # print("responsibilities: ", responsibilities.shape)
        for k in range(self.n_components):
            weight = np.mean(responsibilities[:, k])
            mean = np.sum(responsibilities[:, k][:, np.newaxis] * X, axis=0) / np.sum(responsibilities[:, k])
            covariance = np.dot((responsibilities[:, k][:, np.newaxis] * (X - mean)).T, (X - mean)) / np.sum(responsibilities[:, k])
            self.weights[k] = weight
            self.means[k] = mean
            self.covariances[k] = covariance

    def __pdf(self, X, mean, covariance):
        """
        function to compute the pdf of a multivariate gaussian distribution
        :param X: feature matrix
        :param mean: mean of the distribution
        :param covariance: covariance of the distribution
        :return:
        """
        if np.linalg.det(covariance) < 1e-9:
            covariance = covariance + 1e-8 * np.eye(covariance.shape[0])

        return (1. / ((2 * np.pi)**0.5 * np.linalg.det(covariance)**0.5) * np.exp(-0.5 * np.sum(np.dot(X - mean, np.linalg.inv(covariance)) * (X - mean), axis=1)))



# responsibilities[:, k] = self.weights[k] * multivariate_normal.pdf(X, mean=self.means[k], cov=self.covariances[k])