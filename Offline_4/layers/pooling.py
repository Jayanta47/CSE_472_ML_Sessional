import sys
sys.path.append('/home/jayanta/Documents/CSE_472_Assignments/Offline_4/')

import numpy as np
from numpy.lib.stride_tricks import as_strided

from utils.config import get_layer_num, increment_layer_num

def pool(A, kernel_size, stride, padding=0, pool_mode='max'):

    (N, C, H, W) = A.shape
    # Padding
    A = np.pad(A, pad_width=((0,), (0,), (padding, ), (padding, )), mode='constant', constant_values=(0.,))

    # Window view of A
    output_shape = (N, C, (H - kernel_size + 2*padding) // stride + 1,
                    (W - kernel_size + 2*padding) // stride + 1)
    
    batch_str, channel_str, kern_h_str, kern_w_str = A.strides

    # shape_w = (output_shape[0], output_shape[1], kernel_size, kernel_size)
    shape_w = (N, C, output_shape[2], output_shape[3], kernel_size, kernel_size)
    # strides_w = (stride*A.strides[0], stride*A.strides[1], A.strides[0], A.strides[1])
    strides_w = (batch_str, channel_str, stride * kern_h_str, stride * kern_w_str, kern_h_str, kern_w_str)

    A_w = as_strided(A, shape_w, strides_w)

    # Return the result of pooling
    if pool_mode == 'max':
        return A_w.max(axis=(4, 5))
    elif pool_mode == 'avg':
        return A_w.mean(axis=(4, 5))

# 


class Pooling:
    def __init__(self, kernel_shape=(3, 3), stride=1, mode="max", name=None):
        '''
        :param kernel_shape:
        :param stride:
        :param mode:
        '''
        self.params = {
            'kernel_shape': kernel_shape,
            'stride': stride,
            'mode': mode
        }
        self.type = 'pooling'
        self.cache = {}
        self.has_units = False
        self.name = name

    def has_weights(self):
        return self.has_units

    def forward(self, X, save_cache=False):
        '''
        N: number of data
        C: number of channels
        H: height
        W: width

        '''
        if self.name is None:
            self.name = '{}_{}'.format(self.type, get_layer_num(self.type))
            increment_layer_num(self.type)
        
        N, C, H, W = X.shape
        filter_shape_h, filter_shape_w = self.params['kernel_shape']

        out_h = int(1 + (H - filter_shape_h) / self.params['stride'])
        out_w = int(1 + (W - filter_shape_w) / self.params['stride'])
        out_c = C


        A = pool(X, filter_shape_h, self.params['stride'], padding=0, pool_mode=self.params['mode'])
        if save_cache:
            self.cache['A'] = X

        return A 

    def distribute_value(self, dz, shape):
        (n_H, n_W) = shape
        average = 1 / (n_H * n_W)
        return np.ones(shape) * dz * average

    def create_mask(self, x):
        return x == np.max(x)

    def backward(self, dA):
        print("Backprop of Pooling Layer: ", self.name)
        A = self.cache['A']
        filter_shape_h, filter_shape_w = self.params['kernel_shape']

        (num_data_points, prev_channels, prev_height, prev_width) = A.shape
        m, n_C, n_H, n_W = dA.shape

        dA_prev = np.zeros(shape=(num_data_points, prev_channels, prev_height, prev_width))

        for i in range(num_data_points):
            a = A[i]

            for c in range(n_C):

                for h in range(n_H):
                    for w in range(n_W):

                        vert_start = h * self.params['stride']
                        vert_end = vert_start + filter_shape_h
                        horiz_start = w * self.params['stride']
                        horiz_end = horiz_start + filter_shape_w

                        if self.params['mode'] == 'average':
                            da = dA[i, c, h, w]
                            dA_prev[i, c, vert_start: vert_end, horiz_start: horiz_end] += \
                                self.distribute_value(da, self.params['kernel_shape'])

                        else:
                            a_slice = a[c, vert_start: vert_end, horiz_start: horiz_end]
                            mask = self.create_mask(a_slice)
                            dA_prev[i, c, vert_start: vert_end, horiz_start: horiz_end] += \
                                dA[i, c, h, w] * mask

        return dA_prev






if __name__ == "__main__":
    x = np.random.rand(2,96,7,7)
    pl = Pooling(kernel_shape=(3,3), stride=2, mode="max")
    y = pl.forward(x)
    print(y.shape)




# def pool(A, kernel_size, stride, padding=0, pool_mode='max'):
#     """Perform 2D pooling on the input array."""
   

#     (N, C, H, W) = A.shape
#     # Padding
#     A = np.pad(A, pad_width=((0,), (0,), (padding, ), (padding, )), mode='constant', constant_values=(0.,))

#     # Window view of A
#     output_shape = (N, C, (A.shape[0] - kernel_size) // stride + 1,
#                     (A.shape[1] - kernel_size) // stride + 1)
    
#     batch_str, channel_str, kern_h_str, kern_w_str = A.strides

#     # shape_w = (output_shape[0], output_shape[1], kernel_size, kernel_size)
#     shape_w = (N, C, output_shape[0], output_shape[1], kernel_size, kernel_size)
#     # strides_w = (stride*A.strides[0], stride*A.strides[1], A.strides[0], A.strides[1])
#     strides_w = (batch_str, channel_str, stride * kern_h_str, stride * kern_w_str, kern_h_str, kern_w_str)

#     A_w = as_strided(A, shape_w, strides_w)

#     # Return the result of pooling
#     if pool_mode == 'max':
#         return A_w.max(axis=(2, 3))
#     elif pool_mode == 'avg':
#         return A_w.mean(axis=(2, 3))



# def forward_propagate(X, params, save_cache=False):
#         '''
#         :param X:
#         :param save_cache:
#         :return:
#         '''

#         (N, C, H, W) = X.shape
#         filter_shape_h, filter_shape_w = params['kernel_shape']

#         n_H = int(1 + (H - filter_shape_h) / params['stride'])
#         n_W = int(1 + (W - filter_shape_w) / params['stride'])
#         n_C = C

#         A = np.zeros((N,n_C, n_H, n_W))

#         for i in range(N):
#             for c in range(n_C):
#                 for h in range(n_H):
#                     for w in range(n_W):

#                         vert_start = h * params['stride']
#                         vert_end = vert_start + filter_shape_h
#                         horiz_start = w * params['stride']
#                         horiz_end = horiz_start + filter_shape_w
#                         if params['mode'] == 'average':
#                             A[i, c, h, w] = np.mean(X[i, c, vert_start: vert_end, horiz_start: horiz_end])
#                         else:
#                             A[i, c, h, w] = np.max(X[i, c, vert_start: vert_end, horiz_start: horiz_end])
#         # if save_cache:
#         #     self.cache['A'] = X

#         return A



