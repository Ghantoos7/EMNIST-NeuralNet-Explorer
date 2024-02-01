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


model = Sequential()

model.add(Dense(64, input_dim=784))
model.add(Activation('relu'))

model.add(Dense(64))
model.add(Activation('relu'))

model.add(Dense(10))
model.add(Activation('softmax'))


DEFAULT_LEARNING_RATE = 0.01
DEFAULT_NUMBER_HIDDEN = 2
DEFAULT_BATCH_SIZE = 32

scores = {}

for batch_num in [1,2,4,8,16,32,64,128]:
     for number_hidden in [1,2,4,6,8]:
          for learning_rate in [0.01,0.05,0.1,0.2,0.4,0.8]:
               
               
                        

        

        

        


np.save('scores.npy', scores)


    
