from layers.activation import Relu
from layers.convolution import Convolution
from layers.flatten import Flatten
from layers.fully_connected import FullyConnected
from layers.pooling import Pooling
from layers.softmax import Softmax

from utils.dataLoader import getData
from model.model import Model

from loss.categoricalCrossEntropy import CategoricalCrossEntropy

import numpy as np

np.random.seed(42)

def main():
    image_shape = (64, 64)
    X_train, Y_train = getData(filepath="./dummy_data/sample_train.csv", 
                               folder_path='./dummy_data/train', image_shape=image_shape)
    X_test, Y_test = getData(filepath="./dummy_data/sample_val.csv", 
                            folder_path='./dummy_data/validation', image_shape=image_shape)

    X_train = X_train / 255.0
    X_test = X_test / 255.0

    print("Train data shape: {}, {}".format(X_train.shape, Y_train.shape))
    print("Test data shape: {}, {}".format(X_test.shape, Y_test.shape))

    # AlexNet Architecture 
    # Convolution (11*11*96, stride->4) -> Relu -> 
    # Pooling (3*3, stride->2)-> 
    # Convolution (5*5*256, stride->2)-> Relu -> 
    # Pooling (3*3, stride->2)-> 
    # Convolution (3*3*384, pooling->1)-> Relu -> 
    # Convolution (3*3*384, pooling->1)-> Relu -> 
    # Convolution (3*3*256, pooling->1)-> Relu -> 
    # Pooling (3*3, stride->2)-> 
    # Flatten -> 
    # FullyConnected (4096)-> Relu -> 
    # FullyConnected (4096)-> Relu -> 
    # FullyConnected (1000)-> 
    # Softmax
    '''
    My Architecture
    --------------------------------
    Convolution (12*12*32, stride->3) -> Relu ->
    Pooling (3*3, stride->2)->
    Convolution (4*4*96, stride->2, padding->2)-> Relu ->
    Pooling (3*3, stride->2)->
    Convolution (3*3*128, stride->1, padding->1)-> Relu ->
    Convolution (3*3*192, stride->1, padding->1)-> Relu ->
    Convolution (3*3*96, stride->1, padding->1)-> Relu ->
    Pooling (3*3, stride->2)->
    Flatten -> (864)
    FullyConnected (512)-> Relu ->
    FullyConnected (512)-> Relu ->
    FullyConnected (192)-> Relu ->
    FullyConnected (10)->
    Softmax
    '''

    # model = Model(
    #     Convolution(filters=32, kernel_shape=(8, 8), stride=3, padding=0, name="conv1"),
    #     Relu(),
    #     Pooling(kernel_shape=(3, 3), stride=2, mode = "max", name="pool1"),
    #     Convolution(filters=96, kernel_shape=(4, 4), stride=2, padding=2, name="conv2"),
    #     Relu(),
    #     Pooling(kernel_shape=(3, 3), stride=2, mode = "max", name="pool2"),
    #     Convolution(filters=128, kernel_shape=(3, 3), stride=1, padding=1, name="conv3"),
    #     Relu(),
    #     Convolution(filters=192, kernel_shape=(3, 3), stride=1, padding=1, name="conv4"),
    #     Relu(),
    #     Convolution(filters=96, kernel_shape=(3, 3), stride=1, padding=1, name="conv5"),
    #     Relu(),
    #     Pooling(kernel_shape=(3, 3), stride=2, mode = "max", name="pool3"),
    #     Flatten(),
    #     FullyConnected(256),
    #     Relu(),
    #     FullyConnected(256),
    #     Relu(),
    #     FullyConnected(128),
    #     Relu(),
    #     FullyConnected(10),
    #     Softmax(),
    #     name='cnnALEX'
    # )   

    model = Model(
        Convolution(filters=16, kernel_shape=(16, 16),stride=2, padding=0, name="conv1"),
        Relu(),
        Pooling(kernel_shape=(2, 2), stride=2,mode='max', name="pool1"),
        Convolution(filters=32, kernel_shape=(3, 3), stride=1, padding=1, name="conv2"),
        Relu(),
        Convolution(filters=64, kernel_shape=(3, 3), stride=1, padding=1, name="conv3"),
        Relu(),
        Pooling(kernel_shape=(2, 2), stride=2, mode = "max", name="pool3"),
        Flatten(),
        FullyConnected(units =256),
        Relu(),
        FullyConnected(units =128),
        Relu(),
        # FullyConnected(units= 32),
        # Relu(),
        FullyConnected(units = 10),
        Softmax(),
        name='cnnALEXmini'
    )

    
    
    model.set_loss(CategoricalCrossEntropy)
    # model.set_batch_size(64)
    model.train(X_train, Y_train, epochs=2, batch_size=64)

    print("Testing Accuracy: {}, Macro-f1: {}".format(*model.evaluate(X_test, Y_test)))


if __name__ == "__main__":
    main()




