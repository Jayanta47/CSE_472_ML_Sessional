import numpy as np

class Softmax:
    def __init__(self):
        self.cache = {}
        self.has_units = False

    def has_weights(self):
        return self.has_units

    def forward(self, Z, save_cache=False):
        if save_cache:
            self.cache['Z'] = Z
        Z_ = Z - Z.max()
        e = np.exp(Z_)
        return e / np.sum(e, axis=0, keepdims=True)

    def backward(self, dA):
        Z = self.cache['Z']
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