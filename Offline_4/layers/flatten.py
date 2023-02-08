import numpy as np

class Flatten:
    def __init__(self, transpose=True):
        self.shape = ()
        self.transpose = transpose
        self.has_units = False

    def has_weights(self):
        return self.has_units

    def forward(self, Z, save_cache=False):
        shape = Z.shape
        if save_cache:
            self.shape = shape
        data = np.ravel(Z).reshape(shape[0], -1)
        if self.transpose:
            data = data.T
        return data

    def backward(self, Z):
        if self.transpose:
            Z = Z.T
        return Z.reshape(self.shape)
    

if __name__ == '__main__':
    flatten = Flatten()
    x = np.random.rand(2,96,3,3)
    print(x.shape)
    print(flatten.forward(x, save_cache=True).shape)
    print(flatten.backward(x).shape)

    x = np.random.rand(2,1,3,3)
    print(x)
    y = flatten.forward(x, save_cache=True)
    print(y)
    print(flatten.backward(y))