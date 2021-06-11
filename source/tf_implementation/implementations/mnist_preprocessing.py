import tensorflow

from tf_implementation.interfaces.i_mnist_data import IMNISTData
from tf_implementation.interfaces.i_mnist_preprocessing import IMNISTPreprocessing


class MNISTPreprocessing(IMNISTPreprocessing, object):
    def __init__(self):
        self._new_shape = (-1, 28, 28, 1)
        self._new_type = 'float32'
        self._normalization_factor = 255.

    def apply(self, mnist_data: IMNISTData):
        if not IMNISTData.provided_by(mnist_data):
            raise TypeError("The data provided does not inherit from IMNISTData.")

        mnist_data.training_input = mnist_data.training_input.reshape(self._new_shape).astype(
            self._new_type) / self._normalization_factor

        mnist_data.test_input = mnist_data.test_input.reshape(self._new_shape).astype(
            self._new_type) / self._normalization_factor

        mnist_data.training_labels = tensorflow.keras.utils.to_categorical(
            mnist_data.training_labels.astype(self._new_type))
        mnist_data.test_labels = tensorflow.keras.utils.to_categorical(
            mnist_data.test_labels.astype(self._new_type))

        return mnist_data
