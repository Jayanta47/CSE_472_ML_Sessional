import numpy as np

def pad_inputs(X, pad):
    '''
    Function to apply zero padding to the image
    :param X:[numpy array]: Dataset of shape (m, height, width, depth)
    :param pad:[int]: number of columns to pad
    :return:[numpy array]: padded dataset
    '''
    return np.pad(X, ((0, 0), (pad[0], pad[0]), (pad[1], pad[1]), (0, 0)), 'constant')


# def get_batches(data, labels, batch_size=512, shuffle=True):
#      pass


def evaluate(labels, predictions):
    '''
    A function to compute the accuracy of the predictions on a scale of 0-1.
    :param labels:[numpy array]: Training labels (or testing/validation if available)
    :param predictions:[numpy array]: Predicted labels
    :return:[float]: a number between [0, 1] denoting the accuracy of the prediction
    '''
    return np.mean(np.argmax(labels, axis=0) == np.argmax(predictions, axis=0))


def get_batches(data, labels, batch_size=256, shuffle=True):
    '''
    Function to get data in batches.
    :param data:[numpy array]: training or test data. 
        Assumes shape=[N, C, H, W] where N is the number of samples
    :param labels:[numpy array]: actual labels corresponding to the data.
        Assumes shape=[N, K] where K is number of classes/results per sample and N is number of samples.
    :param batch_size:[int, Default = 256]: required size of batch. 
    :param shuffle:[boolean, Default = True]: if true, function will shuffle the data
    :return:[numpy array, numpy array]: batch data and corresponding labels
        Dimensions: batch_data = [batch_size, C, H, W] and batch_labels = [batch_size, K]
    '''
    # assert data != None
    # assert labels != None
    print(data.shape, labels.shape)
    assert data.shape[0] == labels.shape[0]
    
    N = data.shape[0]
    num_batches = N//batch_size
    if shuffle:
        shuffled_indices = np.random.permutation(N)
        data = data[shuffled_indices]
        labels = labels[shuffled_indices]
    if num_batches == 0:
        yield (data, labels)
    for batch_num in range(num_batches):
        yield (data[batch_num*batch_size:(batch_num+1)*batch_size],
               labels[batch_num*batch_size:(batch_num+1)*batch_size])
    if N%batch_size != 0 and num_batches != 0:
        yield (data[num_batches*batch_size:], labels[num_batches*batch_size:])


if __name__ == "__main__":
    evaluate()

