import tensorflow


class CapsuleModel():
    def __init__(self, input_shape, number_of_classes, number_of_routings, batch_size):
        self._input_shape = input_shape
        self._number_of_classes = number_of_classes
        self._number_of_routings = number_of_routings
        self._batch_size = batch_size

    def model(self):
        pass
