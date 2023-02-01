import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

data_folder = "./data/"
dataset = "data2D.txt"

def load_dataset(filepath=dataset):
    """
    function for reading data from csv
    and processing to return a 2D feature matrix and a vector of class
    :return:
    """
    X = np.loadtxt(data_folder + filepath, dtype=np.float32)

    return X 


def dimensionReducer(X, mu=None, cov=None, dim=2):
    """
    function for reducing the dimension of the data
    :param X: 2D feature matrix
    :param dim: dimension to reduce to
    :return: reduced feature matrix
    """
    K = mu.shape[0]
    pca = PCA(n_components=dim)
    X = pca.fit_transform(X)
    if mu is None:
        return X
    mu_ = np.zeros((K, 2))
    for i in range(K):
        mu_[i] = np.dot(pca.components_, (mu[i]-pca.mean_))
    
    cov_ = np.zeros((K, 2, 2))
    for i in range(K):
        cov_[i] = np.dot(pca.components_, np.dot(cov[i], pca.components_.T))
    return X, mu_, cov_


