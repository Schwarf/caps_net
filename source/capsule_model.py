import tensorflow

from i_caspule_model import ICapsuleModel
from primary_capsule_layer import PrimaryCapsuleLayer
from capsule_layer import KerasLayerWithWeights

tensorflow.keras.backend.set_image_data_format("channels_last")


class CapsuleModel(ICapsuleModel):

    def __init__(self, input_shape, number_of_classes, number_of_routings, batch_size):
        self._input_shape = input_shape
        self._batch_size = batch_size
        self._primary_capsule_layer = PrimaryCapsuleLayer(dimension_of_capsule=8,
                                                          number_of_channels=32,
                                                          kernel_size=9,
                                                          strides=2,
                                                          padding='valid')

        self._digit_capsule_layer = KerasLayerWithWeights(number_of_capsules=number_of_classes,
                                                          dimension_of_capsule=16,
                                                          number_of_routings=number_of_routings)

    def model(self):
        input_layer = tensorflow.keras.layers.Input(shape=self._input_shape, batch_size=self._batch_size)

        first_convolutional_layer = tensorflow.keras.Conv2D(filters=256, kernel_size=9, strides=1, padding='valid',
                                                   activation='relu', name='conv1')(input_layer)

        primary_capsule_layer = self._primary_capsule_layer.apply(first_convolutional_layer)

        digit_capsule_layer = self._digit_capsule_layer(primary_capsule_layer)
