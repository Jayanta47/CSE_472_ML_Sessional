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
from sklearn.metrics import f1_score

def macro_f1(y_true, y_pred):
    num_classes = y_true.shape[1]
    f1_scores = []
    for c in range(num_classes):
        f1_scores.append(f1_score(y_true[:, c], y_pred[:, c],zero_division='warn'))
    macro_f1 = np.mean(f1_scores)
    return macro_f1

def to_binary_indicator(y):
    # Get the number of classes and number of samples
    num_classes, num_samples = y.shape

    # Create an empty array to store the binary indicators
    y_binary = np.zeros((num_samples, num_classes), dtype=int)

    # For each sample, set the corresponding class to 1
    for i in range(num_samples):
        y_binary[i, np.argmax(y[:, i])] = 1
        
    return y_binary




def evaluate(labels, predictions):
    '''
    A function to compute the accuracy of the predictions on a scale of 0-1.
    :param labels:[numpy array]: Training labels (or testing/validation if available)
    :param predictions:[numpy array]: Predicted labels
    :return:[float]: a number between [0, 1] denoting the accuracy of the prediction
    '''
    # print("In eval func, Label: ", labels.shape,"Predictions: ",  predictions.shape)
    return np.mean(np.argmax(labels, axis=0) == np.argmax(predictions, axis=0)), macro_f1(to_binary_indicator(labels), to_binary_indicator(predictions))


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
    # print(data.shape, labels.shape)
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

def getTestBatches(data, batch_size=256):
    '''
    Function to get data in batches.
    :param data:[numpy array]: training or test data. 
        Assumes shape=[N, C, H, W] where N is the number of samples
    :param batch_size:[int, Default = 256]: required size of batch. 
    :return:[numpy array]: batch data
        Dimensions: batch_data = [batch_size, C, H, W]
    '''
    # assert data != None
    print(data.shape)
    N = data.shape[0]
    num_batches = N//batch_size
    if num_batches == 0:
        yield data
    for batch_num in range(num_batches):
        yield data[batch_num*batch_size:(batch_num+1)*batch_size]
    if N%batch_size != 0 and num_batches != 0:
        yield data[num_batches*batch_size:]

if __name__ == "__main__":
    evaluate()

