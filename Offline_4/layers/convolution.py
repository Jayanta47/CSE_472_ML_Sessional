import sys
sys.path.append('/home/jayanta/Documents/CSE_472_Assignments/Offline_4/')
import numpy as np
import pickle
from os import path, makedirs, remove

from utils.initializers import glorot_uniform
from utils.config import get_layer_num, increment_layer_num

def getWindows(input, output_size, kernel_shape, padding_shape, stride=1, dilate=0):
    kernel_size = kernel_shape[0]
    working_pad = padding_shape[0]

    working_input = input
    # dilate the input if necessary
    # The input is of the shape (batch, channel, height, width) => (N, C, H, W)
    N, C, H, W = input.shape
    if dilate != 0:
        # print("dilate")
        for i in range(1, dilate+1):
            working_input = np.insert(working_input, range(1, i*H, i), 0, axis=2)
            working_input = np.insert(working_input, range(1, i*W, i), 0, axis=3)
        # print("working_input shape: ", working_input.shape)
        # print(working_input)
    # pad the input if necessary
    if working_pad != 0:
        working_input = np.pad(working_input, 
                                pad_width=((0,), (0,), (working_pad,), (working_pad,)), 
                                mode='constant', 
                                constant_values=(0.,))

    _, _, out_h, out_w = output_size
    out_b, out_c, _, _ = input.shape
    batch_str, channel_str, kern_h_str, kern_w_str = working_input.strides

    return np.lib.stride_tricks.as_strided(
        working_input,
        (out_b, out_c, out_h, out_w, kernel_size, kernel_size),
        (batch_str, channel_str, stride * kern_h_str, stride * kern_w_str, kern_h_str, kern_w_str)
    )


class Convolution:
    def __init__(self, filters, kernel_shape=(3, 3), padding=0, stride=1, name=None):
        self.params = {
            'filters': filters,
            'padding': padding,
            'kernel_shape': kernel_shape,
            'stride': stride
        }
        self.cache = {}
        self.rmsprop_cache = {}
        self.momentum_cache = {}
        self.grads = {}
        self.has_units = True
        self.name = name
        self.type = 'conv'

    def has_weights(self):
        return self.has_units

    def save_weights(self, dump_path):
        dump_cache = {
            'cache': self.cache,
            'grads': self.grads,
            'momentum': self.momentum_cache,
            'rmsprop': self.rmsprop_cache
        }
        save_path = path.join(dump_path, self.name+'.pkl')
        makedirs(path.dirname(save_path), exist_ok=True)
        remove(save_path)
        with open(save_path, 'wb') as d:
            pickle.dump(dump_cache, d)

    def load_weights(self, dump_path):
        if self.name is None:
            self.name = '{}_{}'.format(self.type, get_layer_num(self.type))
            increment_layer_num(self.type)
        read_path = path.join(dump_path, self.name+'.pkl')
        with open(read_path, 'rb') as r:
            dump_cache = pickle.load(r)
        self.cache = dump_cache['cache']
        self.grads = dump_cache['grads']
        self.momentum_cache = dump_cache['momentum']
        self.rmsprop_cache = dump_cache['rmsprop']
    

    def forward(self, X, save_cache=False):
        """
        The forward pass of convolution
        :param x: input data of shape (N, C, H, W)
        :return: output data of shape (N, self.out_channels, H', W') where H' and W' are determined by the convolution
                 parameters.
        """
        if self.name is None:
            self.name = '{}_{}'.format(self.type, get_layer_num(self.type))
            increment_layer_num(self.type)

        N, C, H, W = X.shape
        filter_shape_h, filter_shape_w = self.params['kernel_shape']
        
        if 'W' not in self.params:
            # shape = (filter_shape_h, filter_shape_w, C, self.params['filters'])
            shape = (self.params['filters'], C, filter_shape_h, filter_shape_w)
            self.params['W'], self.params['b'] = glorot_uniform(shape=shape)

        # if self.params['padding'] == 'same':
        #     pad_h = int(((H - 1)*self.params['stride'] + filter_shape_h - H) / 2)
        #     pad_w = int(((W - 1)*self.params['stride'] + filter_shape_w - W) / 2)
        #     n_H = H
        #     n_W = W
        # else:
        #     pad_h = 0
        #     pad_w = 0
        #     n_H = int((H - filter_shape_h) / self.params['stride']) + 1
        #     n_W = int((W - filter_shape_w) / self.params['stride']) + 1

        if self.params['padding'] != None:
            pad_h = self.params['padding']
            pad_w = self.params['padding']
            n_H = (H - filter_shape_h + 2 * pad_h) // self.params['stride'] + 1
            n_W = (W - filter_shape_w + 2 * pad_w) // self.params['stride'] + 1
        else:
            pad_h = 0
            pad_w = 0
            n_H = (H - filter_shape_h) // self.params['stride'] + 1
            n_W = (W - filter_shape_w) // self.params['stride'] + 1

        self.params['pad_h'], self.params['pad_w'] = pad_h, pad_w
        
        padding = pad_h
        stride = self.params['stride']

        out_h = n_H
        out_w = n_W

        out_h = (H - filter_shape_h + 2 * padding) // stride + 1
        out_w = (W - filter_shape_w + 2 * padding) // stride + 1

        
        
        windows = getWindows(X, (N, C, out_h, out_w), 
                             (filter_shape_h, filter_shape_w), 
                             (pad_h, pad_w), self.params['stride'])
        # print(windows.shape)
        # print(self.params['W'].shape)
        # print(self.params['b'].shape)
        # return 
        out = np.einsum('bihwkl,oikl->bohw', windows, self.params['W'])
        # print(out.shape)
        # add bias to kernels   
        out += self.params['b']

        self.cache['X'] = X
        self.cache['wd'] = windows 

        return out

    def backward(self, dZ):
        '''
        :param dZ:
        :return:
        '''
        print("Backprop of Conv Layer: ", self.name)
        X = self.cache['X']
        windows = self.cache['wd']
        filter_shape_h, filter_shape_w = self.params['kernel_shape']
        pad_h, pad_w = self.params['pad_h'], self.params['pad_w']

        self.grads = self.init_cache()

        if pad_h == 0:
            pad_h = filter_shape_h - 1
            pad_w = filter_shape_w - 1

        dZ_windows = getWindows(dZ, X.shape, (filter_shape_h, filter_shape_w), 
                             padding_shape = (pad_h, pad_w), stride=1, dilate=self.params['stride'] - 1)
        
        rot_kern = np.rot90(self.params['W'], 2, axes=(2, 3)) 
        # print("is null", np.isnan(dZ_windows).any())
        # print("windows shape: ", windows.shape)
        # print("dZ shape: ", dZ_windows.shape)
        # print(rot_kern.shape)
        
        db = np.sum(dZ, axis=(0, 2, 3))
        
        dw = np.einsum('bihwkl,bohw->oikl', windows, dZ)
        
        dX = np.einsum('bohwkl,oikl->bihw', dZ_windows, rot_kern)
        # print("haha")
        self.grads['dW'] = dw
        self.grads['db'] = db.reshape(self.params['b'].shape) 
        # print(db.shape)
        
        return dX 


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
        # print("newDB", new_db.shape)
        # print("grads_db",self.grads['db'].shape )
        # print("rms db", self.rmsprop_cache['db'].shape)

        if amsprop:
            self.rmsprop_cache['dW'] = np.maximum(self.rmsprop_cache['dW'], new_dW)
            self.rmsprop_cache['db'] = np.maximum(self.rmsprop_cache['db'], new_db)
        else:
            self.rmsprop_cache['dW'] = new_dW
            self.rmsprop_cache['db'] = new_db

        # print("rms", self.rmsprop_cache['db'].shape)

    def apply_grads(self, learning_rate=0.01, l2_penalty=1e-4, optimization='adam', epsilon=1e-8,
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

    # conv = Convolution(filters = 32, kernel_shape=(8,8), stride=3, padding=0)
    # X = np.random.randn(2, 3, 128, 128)
    # out = conv.forward(X)
    # print(out.shape)
    # out_b = conv.backward(out)
    # print(out_b.shape)

    conv = Convolution(filters = 32, kernel_shape=(3,3), stride=3, padding=0)
    X = np.random.randn(2, 3, 9, 9)
    out = conv.forward(X)
    print(out.shape)
    out_b = conv.backward(out)
    print(out_b.shape)
    conv.momentum()
    conv.rmsprop()
    conv.apply_grads()