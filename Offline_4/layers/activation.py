import numpy as np

# create a class of Relu Function with numpy
class Relu:
    def __init__(self):
        self.mask = None
        self.has_units = False

    def has_weights(self):
        return self.has_units

    def forward(self, x, save_cache=False):

        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0

        # if save_cache:
        #     self.cache['X'] = x
        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout

        return dx
    

if __name__ == '__main__':
    x = np.array([[-1.0, -2.0, 3.0], [-4.0, 5.0, 6.0]])
    print(x)
    relu = Relu()
    print(relu.forward(x))
    print(relu.backward(x))

    x = np.random.rand(2,3,57,57)
    print(x.shape)
    print(relu.forward(x).shape)
    print(relu.backward(x).shape)
    