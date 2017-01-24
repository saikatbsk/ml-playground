### MNIST-CNN

CNN model trained to classify handwritten digits from the MNIST dataset. MNIST is a dataset of 60,000 28x28 grayscale images of the 10 digits, along with a test set of 10,000 images.

Requirements are,

* Python 3.x
* Numpy
* Matplotlib
* Keras
* TensorFlow

The first 50,000 images, from the training dataset, are used for training. The last 10,000 images are used for validation. Accuracy on testset after 20 epochs is **99.16 %**.

Accuracy and loss for each epoch is shown in Figure 1.

|![figure_1](figure_1.png)|
|---|
|<small>Figure 1. Model accuracy and model loss for 20 epochs.</small>|
