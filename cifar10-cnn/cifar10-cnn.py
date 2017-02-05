from __future__ import print_function
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from keras.datasets import cifar10
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D

""" Initialize some variables. """
nb_classes = 10
nb_epoch = 50
batch_size = 32
pool_size = (2, 2)      # Size of poolig area
kernel_size = (3, 3)    # Convolution kernel size

""" Load CIFAR10, and split into training and test sets. """
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
print('X_train shape: ', X_train.shape)
print(X_train.shape[0], 'training samples')
print(X_test.shape[0], 'test samples')

""" Change type and normalize. """
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

"""
Convert class vectors to binary class matrices using the 1-hot encoding method.
"""
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

""" Create a sequential model. """
model = Sequential()

model.add(Convolution2D(32, kernel_size[0], kernel_size[1],
                        border_mode='same', input_shape=X_train.shape[1:]))
model.add(Activation('relu'))
model.add(Convolution2D(32, kernel_size[0], kernel_size[1]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=pool_size))
model.add(Dropout(0.25))

model.add(Convolution2D(64, kernel_size[0], kernel_size[1], border_mode='same'))
model.add(Activation('relu'))
model.add(Convolution2D(64, kernel_size[0], kernel_size[1]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=pool_size))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

history = model.fit(X_train, Y_train,
                    batch_size=batch_size,
                    nb_epoch=nb_epoch,
                    verbose=1,
                    validation_data=(X_test, Y_test),
                    shuffle=True)

""" Save trained weights for future use """
model.save_weights('cifar10_cnn_weights.h5')

fig = plt.figure(figsize=(14, 6))
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

#plt.show()
fig.savefig('fig/acc_loss.png')

score = model.evaluate(X_test, Y_test, verbose=0)
