from __future__ import print_function

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop

""" Starting with a random seed ensures the reproducibility of the tests. """
np.random.seed(1337)

""" Initialize some variables. """
no_classes = 10
no_epoch = 20
batch_size = 128

"""
The MNIST dataset is provided with Keras. MNIST is a dataset of 60,000 28x28
grayscale images of the 10 digits, along with a test set of 10,000 images.
"""
(X_train, y_train), (X_test, y_test) = mnist.load_data()

""" Width and height of the training images. """
img_width = X_train.shape[1]
img_height = X_train.shape[2]

""" Let's check some of the sample images. """
fig = plt.figure()
gs = gridspec.GridSpec(2, 5)

for i in range(10):
    idx = np.random.randint(1000)
    sample_img = X_train[idx, :, :]
    sample_label = y_train[idx]

    fig.add_subplot(gs[i])
    plt.imshow(sample_img, cmap='gray')
    plt.title(sample_label)

plt.show()

"""
The dataset needs to be reshaped as each 28x28 image needs to be represented
using a single vector.
"""
X_train = X_train.reshape(X_train.shape[0], img_width * img_height)
X_test = X_test.reshape(X_test.shape[0], img_width * img_height)

""" Change type and normalize. """
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train /= 255
X_test /= 255

print(X_train.shape[0], 'train samples.')
print(X_test.shape[0], 'test samples.')

"""
Convert class vectors to binary class matrices using the 1-hot encoding method.
"""
Y_train = np_utils.to_categorical(y_train, no_classes)
Y_test = np_utils.to_categorical(y_test, no_classes)

""" Create a sequential model. """
model = Sequential()

model.add(Dense(512, input_shape=(img_width*img_height,)))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(10))
model.add(Activation('softmax'))

""" Let's look at the summary of the model. """
model.summary()

model.compile(loss='categorical_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])

history = model.fit(X_train, Y_train,
                    batch_size=batch_size,
                    nb_epoch=no_epoch,
                    verbose=1,
                    validation_data=(X_test, Y_test))

score = model.evaluate(X_test, Y_test, verbose=0)

print('Test score: ', score[0])
print('Test accuracy: ', score[1])
