import numpy as np
import sys

np.random.seed(42)  

from keras.models import Sequential
from keras.layers import Dense, Activation
from tensorflow.python.keras.utils import np_utils
from keras.optimizers import SGD
from keras.datasets import mnist
 
# Load pre-shuffled MNIST data into train and test sets
(X_train_1, train_labels_1), (X_test, test_labels_1) = mnist.load_data()
 
# Preprocess input data
X_train_1 = X_train_1.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)
X_train_1 = X_train_1.astype('float32')
X_test = X_test.astype('float32')

X_train_1 /= 255
X_test /= 255

# Divides the dataset into train and validation sets
X_valid = X_train_1[50000:60000]
X_train = X_train_1[:50000]
print(X_train.shape[0], 'train samples')
print(X_valid.shape[0], 'validation samples')
 
# Preprocess class labels
train_labels = np_utils.to_categorical(train_labels_1, 10)
test_labels = np_utils.to_categorical(test_labels_1, 10)
valid_labels = train_labels[50000:60000]
train_labels = train_labels[:50000]

