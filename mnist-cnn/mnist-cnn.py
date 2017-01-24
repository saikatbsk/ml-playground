from __future__ import print_function

import numpy as np

from keras import backend as K

""" Starting with a random seed ensures the reproducibility of the tests. """
np.random.seed(1337)

""" Initialize some variables. """
no_classes = 10
no_epoch = 20
batch_size = 128

no_filter = 32          # Number of convolutional filters to use
pool_size = (2, 2)      # Size of poolig area
kernel_size = (3, 3)    # Convolution kernel size

"""
The MNIST dataset is provided with Keras. MNIST is a dataset of 60,000 28x28
grayscale images of the 10 digits, along with a test set of 10,000 images.
"""
(X_train, y_train), (X_test, y_test) = mnist.load_data()

""" Width and height of the training images. """
img_width = X_train.shape[1]
img_height = X_train.shape[2]

"""
The Convolution2D layers in Keras, are designed to work with 3 dimensions per
example. They have 4-dimensional inputs and outputs. This covers colour images
(nb_samples, nb_channels, width, height), but more importantly, it covers
deeper layers of the network, where each example has become a set of feature
maps i.e. (nb_samples, nb_features, width, height).

The greyscale image for MNIST digits input would either need a different CNN
layer design (or a param to the layer constructor to accept a different shape),
or the design could simply use a standard CNN and you must explicitly express
the examples as 1-channel images.
"""
if K.image_dim_ordering() == 'th':
    X_train = X_train.reshape(X_train.shape[0], 1, img_height, img_width)
    X_test = X_test.reshape(X_test.shape[0], 1, img_height, img_width)
    input_shape = (1, img_height, img_width)
else:
    X_train = X_train.reshape(X_train.shape[0], img_height, img_width, 1)
    X_test = X_test.reshape(X_test.shape[0], img_height, img_width, 1)
    input_shape = (img_height, img_width, 1)

""" Change type and normalize. """
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train /= 255
X_test /= 255

print(X_train.shape[0], 'train samples.')
print(X_test.shape[0], 'test samples.')
