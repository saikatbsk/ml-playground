from __future__ import print_function
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from keras.datasets import cifar10

# Load CIFAR10, and split into training and test sets
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
print('X_train shape: ', X_train.shape)
print(X_train.shape[0], 'training samples')
print(X_test.shape[0], 'test samples')
