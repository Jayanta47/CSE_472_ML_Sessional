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


def dimensionReducer(X, dim=2):
    """
    function for reducing the dimension of the data
    :param X: 2D feature matrix
    :param dim: dimension to reduce to
    :return: reduced feature matrix
    """
    
    pca = PCA(n_components=dim)
    pca.fit(X)
    X = pca.transform(X)
    return X


