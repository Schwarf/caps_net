from interfaces.i_mnist_data import IMNISTData


class MNISTData(IMNISTData, object):
    def __init__(self):
        from tensorflow.keras.datasets import mnist
        (self._training_input, self._training_labels), (self._test_input, self._test_labels) = mnist.load_data()

    @property
    def training_input(self):
        return self._training_input

    @property
    def training_labels(self):
        return self._training_labels

    @property
    def test_input(self):
        return self._test_input

    @property
    def test_labels(self):
        return self._test_labels

    @training_input.setter
    def training_input(self, training_input):
        self._training_input = training_input

    @test_labels.setter
    def test_labels(self, test_labels):
        self._test_labels = test_labels

    @test_input.setter
    def test_input(self, test_input):
        self._test_input = test_input

    @training_labels.setter
    def training_labels(self, training_labels):
        self._training_labels = training_labels
