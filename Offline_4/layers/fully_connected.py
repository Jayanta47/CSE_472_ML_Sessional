import sys
sys.path.append('/home/jayanta/Documents/CSE_472_Assignments/Offline_4/')

import numpy as np
import pickle
from os import path, makedirs, remove 

from utils.config import  get_layer_num, increment_layer_num
from utils.initializers import he_initializer

np.random.seed(0)

class FullyConnected:
    def __init__(self, units = 100, name=None):
        # self.input_size = input_size
        # self.output_size = output_size
        # self.weight_decay_lambda = weight_decay_lambda
        self.params = {}
        self.grads = {}
        self.cache = {}
        self.momentum_cache = {}
        self.rmsprop_cache = {}
        self.type = 'fc'
        self.name = name
        self.units = units
        self.has_units = True

    def has_weights(self):
        return self.has_units

    def save_weights(self, path):
        dump_cache = {
            'cache': self.cache,
            'grads': self.grads,
            'momentum': self.momentum_cache,
            'rms_prop': self.rmsprop_cache,
        }
        save_path = path.join(path, self.name + '.pkl')
        makedirs(path.dirname(save_path), exist_ok=True)
        remove(save_path)
        with open(save_path, 'wb') as f:
            pickle.dump(dump_cache, f)

    def load_weights(self, path):
        if self.name is None:
            self.name = '{}_{}'.format(self.type, get_layer_num(self.type))
            increment_layer_num(self.type)

        path = path.join(path, self.name + '.pkl')
        with open(path, 'rb') as f:
            dump_cache = pickle.load(f)
            self.cache = dump_cache['cache']
            self.grads = dump_cache['grads']
            self.momentum_cache = dump_cache['momentum']
            self.rmsprop_cache = dump_cache['rms_prop']

    def forward(self, x, save_cache=False):
        if self.name is None:
            self.name = '{}_{}'.format(self.type, get_layer_num(self.type))
            increment_layer_num(self.type)
        
        if 'W' not in self.params:
            self.params['W'], self.params['b'] = he_initializer((x.shape[0], self.units))

        W, b = self.params['W'], self.params['b']
        out = np.dot(W, x) + b

        if save_cache:
            self.cache['x'] = x

        return out

    def backward(self, dZ):
        W, b = self.params['W'], self.params['b']
        x = self.cache['x']
        batch_size = dZ.shape[1]

        dx = np.dot(W.T, dZ)
        dW = np.dot(dZ, x.T) / batch_size
        db = np.sum(dZ, axis=1, keepdims=True)

        self.grads['dW'] = dW
        self.grads['db'] = db

        return dx
    
    def init_cache(self):
        cache = dict()
        cache['dW'] = np.zeros_like(self.params['W'])
        cache['db'] = np.zeros_like(self.params['b'])
        return cache
    
    def momentum(self, beta=0.9):
        if not self.momentum_cache:
            self.momentum_cache = self.init_cache()
        self.momentum_cache['dW'] = beta * self.momentum_cache['dW'] + (1 - beta) * self.grads['dW']
        self.momentum_cache['db'] = beta * self.momentum_cache['db'] + (1 - beta) * self.grads['db']


    def rmsprop(self, beta=0.999, amsprop=True):
        if not self.rmsprop_cache:
            self.rmsprop_cache = self.init_cache()

        new_dW = beta * self.rmsprop_cache['dW'] + (1 - beta) * (self.grads['dW']**2)
        new_db = beta * self.rmsprop_cache['db'] + (1 - beta) * (self.grads['db']**2)

        if amsprop:
            self.rmsprop_cache['dW'] = np.maximum(self.rmsprop_cache['dW'], new_dW)
            self.rmsprop_cache['db'] = np.maximum(self.rmsprop_cache['db'], new_db)
        else:
            self.rmsprop_cache['dW'] = new_dW
            self.rmsprop_cache['db'] = new_db
            

    def apply_grads(self, learning_rate=0.001, l2_penalty=1e-4, optimization='adam', epsilon=1e-8, \
                correct_bias=False, beta1=0.9, beta2=0.999, iter=999):
        if optimization != 'adam':
            self.params['W'] -= learning_rate * (self.grads['dW'] + l2_penalty * self.params['W'])
            self.params['b'] -= learning_rate * (self.grads['db'] + l2_penalty * self.params['b'])
        else:
            if correct_bias:
                W_first_moment = self.momentum_cache['dW'] / (1 - beta1 ** iter)
                b_first_moment = self.momentum_cache['db'] / (1 - beta1 ** iter)
                W_second_moment = self.rmsprop_cache['dW'] / (1 - beta2 ** iter)
                b_second_moment = self.rmsprop_cache['db'] / (1 - beta2 ** iter)
            else:
                W_first_moment = self.momentum_cache['dW']
                b_first_moment = self.momentum_cache['db']
                W_second_moment = self.rmsprop_cache['dW']
                b_second_moment = self.rmsprop_cache['db']

            W_learning_rate = learning_rate / (np.sqrt(W_second_moment) + epsilon)
            b_learning_rate = learning_rate / (np.sqrt(b_second_moment) + epsilon)

            self.params['W'] -= W_learning_rate * (W_first_moment + l2_penalty * self.params['W'])
            self.params['b'] -= b_learning_rate * (b_first_moment + l2_penalty * self.params['b'])


if __name__ == "__main__":
    x = np.random.rand(864, 2)
    fc = FullyConnected(units=512)
    out = fc.forward(x)
    print(out.shape)
    print(out)