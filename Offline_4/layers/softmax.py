import numpy as np


import numpy as np

def replace_nan(arr, value):
    is_nan = np.isnan(arr)
    arr[is_nan] = value

    return arr


class Softmax:
    def __init__(self):
        self.cache = {}
        self.has_units = False

    def has_weights(self):
        return self.has_units

    def forward(self, Z, save_cache=False, keep_prob = 1.0):
        if save_cache:
            self.cache['Z'] = Z
        # print(Z.max())
        Z = np.clip(Z, 1e-15, 1. - 1e-15)
        Z_ = Z - np.max(Z, axis=0,keepdims=True)
        # print(Z_)
        e = np.exp(Z_)
        e = replace_nan(e, 1e-15)
        return e / np.sum(e, axis=0, keepdims=True)
        # return np.random.rand(*Z.shape)

    def backward(self, dA):
        Z = self.cache['Z']
        # np.clip(dA, 0, 1e6)
        return dA * (Z * (1 - Z))
    

if __name__ == '__main__':
    softmax = Softmax()
    x = np.random.rand(10, 2)
    print(x.shape)
    print(softmax.forward(x, save_cache=True).shape)
    print(softmax.backward(x).shape)

    x = np.random.rand(2,1,3,3)
    print(x)
    y = softmax.forward(x, save_cache=True)
    print(y)
    print(softmax.backward(y))