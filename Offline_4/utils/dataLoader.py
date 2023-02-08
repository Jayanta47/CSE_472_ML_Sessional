import cv2
import numpy as np
import os
import pandas as pd

TOTAL_BATCHES = 5
NUM_DIMENSIONS = 3072
NUM_CLASSES = 10
SAMPLES_PER_BATCH = 10000
MAX_TRAINING_SAMPLES = 50000
MAX_TESTING_SAMPLES = 10000
FILE_NAME = {
    'training': 'data_batch_',
    'testing': 'test_batch'
}

# load image dataset using opencv
def load_images(path, size=(256,256)):
    data_list = list()
    # enumerate filenames in directory, assume all are images
    for filename in os.listdir(path):
        # load and resize the image
        pixels = cv2.imread(path + filename)
        pixels = cv2.resize(pixels, size)
        # store
        data_list.append(pixels)
    return np.asarray(data_list)

def getData(filepath, folder_path, image_shape = (128, 128),num_samples=100000, dataset='train'):
    df = pd.read_csv(filepath)
  
    data = []
    labels = []
    for index, row in df.iterrows():
        filename = row['filename']
        label = row['label']
        img = cv2.imread(os.path.join(folder_path, filename))
        img = cv2.resize(img, image_shape)
        if img is not None:
            # print(img.shape)
            data.append(img)
            label_ohe = np.zeros(10)
            label_ohe[label] = 1.0
            labels.append(label_ohe)
    data = np.stack(data, axis=0)
    labels = np.array(labels)
    # print(data.shape)
    data = data.transpose((0,3,1,2))
    return data, labels
