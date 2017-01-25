### MNIST-CNN

CNN model trained to classify handwritten digits from the MNIST dataset. MNIST is a dataset of 60,000 28x28 grayscale images of the 10 digits, along with a test set of 10,000 images.

Requirements are,

* Python 2.x (t-sne doesn't work with Python 3.x)
* Pathlib
* Numpy
* Matplotlib
* Graphviz (Optional)
* Pydot (Optional)
* Keras
* TensorFlow

The first 50,000 images, from the training dataset, are used for training. The last 10,000 images are used for validation. Accuracy on testset after 20 epochs is **99.16 %**.

The model visualization is shown in Figure 1.

|![figure_1](figure_1.png)|
|---|
|Figure 1. Model visualization.|

Accuracy and loss for each epoch is shown in Figure 2.

|![figure_2](figure_2.png)|
|---|
|Figure 2. Model accuracy and model loss for 20 epochs.|
