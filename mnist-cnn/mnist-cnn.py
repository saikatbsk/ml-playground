from __future__ import print_function
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from keras import backend as K
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D

""" Optional """
try:
    from keras.utils.visualize_util import plot
except ImportError:
    print('ImportError: Model visualization disabled!')

try:
    from tsne import bh_sne
except ImportError:
    print('ImportError: Data visualization (t-SNE) disabled!')

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

print('Input shape: ', X_train.shape)
print(X_train.shape[0], 'train samples.')
print(X_test.shape[0], 'test samples.')

"""
Convert class vectors to binary class matrices using the 1-hot encoding method.
"""
Y_train = np_utils.to_categorical(y_train, no_classes)
Y_test = np_utils.to_categorical(y_test, no_classes)

""" Data visualization using t-SNE """
if 'bh_sne' in sys.modules:
    print('Visualizing training data using t-SNE...')

    x_data = X_train.astype('float64')
    x_data = x_data.reshape((x_data.shape[0], -1))
    x_data = x_data[:1000]
    y_data = y_train[:1000]

    """ Perform t-SNE embedding """
    vis_data = bh_sne(x_data)

    """ Plot the data """
    vis_x = vis_data[:, 0]
    vis_y = vis_data[:, 1]

    fig = plt.figure()
    plt.scatter(vis_x, vis_y, c=y_data, cmap=plt.cm.get_cmap('jet', 10))
    plt.colorbar(ticks=range(10))
    plt.clim(-0.5, 9.5)
    plt.title('Training data visualization with t-SNE')
    plt.show()
    fig.savefig('fig/data.png')

""" Create a sequential model. """
model = Sequential()

model.add(Convolution2D(no_filter, kernel_size[0], kernel_size[1],
                        border_mode='valid', input_shape=input_shape))
model.add(Activation('relu'))
model.add(Convolution2D(no_filter, kernel_size[0], kernel_size[1]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=pool_size))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(Dense(no_classes))
model.add(Activation('softmax'))

""" Let's look at the summary of the model. """
model.summary()

""" Save model visualization to file """
if 'keras.utils.visualize_util' in sys.modules:
    plot(model, to_file='fig/model.png', show_shapes=True)

model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

weights_file = Path('mnist_cnn_weights.h5')

if weights_file.is_file():
    """ Load pre-computed weights """
    print('Loading weights...')
    model.load_weights(weights_file.name)
else:
    """ Else train the model """
    print('Training model...')
    history = model.fit(X_train, Y_train,
                        batch_size=batch_size,
                        nb_epoch=no_epoch,
                        verbose=1,
                        validation_data=(X_test, Y_test))

    """ Save trained weights for future use """
    model.save_weights(weights_file.name)

    fig = plt.figure()
    gs = gridspec.GridSpec(1, 2)

    """ Summarize history for accuracy """
    fig.add_subplot(gs[0])
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')

    """ Summarize history for loss """
    fig.add_subplot(gs[1])
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')

    plt.show()
    fig.savefig('fig/acc_loss.png')

""" Evaluate the trained model """
score = model.evaluate(X_test, Y_test, verbose=0)

print('Test score: ', score[0])
print('Test accuracy: ', score[1])
