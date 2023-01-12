import pandas as pd
import numpy as np

dataset = "./data_banknote_authentication.csv"

def load_dataset(filepath=dataset):
    """
    function for reading data from csv
    and processing to return a 2D feature matrix and a vector of class
    :return:
    """
    # todo: implement

    df = pd.read_csv(filepath)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    return X, y
    


def split_dataset(X, y, test_size=0.2, shuffle=False):
    """
    function for spliting dataset into train and test
    :param X:
    :param y:
    :param float test_size: the proportion of the dataset to include in the test split
    :param bool shuffle: whether to shuffle the data before splitting
    :return:
    """
    # todo: implement.
    dataset = np.concatenate((X, y.reshape(-1, 1)), axis=1)

    if shuffle:
        np.random.shuffle(dataset)
    train_size = int(len(dataset) * (1 - test_size))
    X_train, y_train = dataset[:train_size, :-1], dataset[:train_size, -1]
    X_test, y_test = dataset[train_size:, :-1], dataset[train_size:, -1]
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
    idx = np.random.choice(X.shape[0], size=X.shape[0], replace=True)
    X_sample = X[idx]
    y_sample = y[idx]
    assert X_sample.shape == X.shape
    assert y_sample.shape == y.shape
    for i in idx:
        assert X[i] in X_sample and y[i] in y_sample
    return X_sample, y_sample

