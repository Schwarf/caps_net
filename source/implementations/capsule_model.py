import numpy
import tensorflow

from implementations.capsule_layer import KerasLayerWithWeights
from implementations.length_layer import LengthLayer
from implementations.mask_layer import MaskLayer
from implementations.primary_capsule_layer import PrimaryCapsuleLayer
from interfaces.i_caspule_model import ICapsuleModel

tensorflow.keras.backend.set_image_data_format("channels_last")


class CapsuleModel(ICapsuleModel, object):

    def __init__(self, input_shape, number_of_classes, number_of_routings, batch_size):
        self._input_shape = input_shape
        self._batch_size = batch_size
        self._number_of_classes = number_of_classes
        self._decoder_activation = tensorflow.keras.activations.relu
        self._primary_capsule_layer = PrimaryCapsuleLayer(dimension_of_capsule=8,
                                                          number_of_channels=32,
                                                          kernel_size=9,
                                                          activation_function=tensorflow.keras.activations.relu,
                                                          strides=2,
                                                          padding='valid')

        self._digit_capsule_layer = KerasLayerWithWeights(number_of_capsules=number_of_classes,
                                                          dimension_of_capsule=16,
                                                          number_of_routings=number_of_routings)
        self._length_layer = LengthLayer()
        self._mask_layer = MaskLayer()
        self._capsule_model = None
        self._decoder = None
        self._input = None

    def _capsule_part(self):
        primary_capsule_layer = self._primary_capsule_layer.apply(self._input)
        digit_capsule_layer = self._digit_capsule_layer(primary_capsule_layer)
        compute_capsules_length = self._length_layer(digit_capsule_layer)
        self._capsule_model = compute_capsules_length

    def _decoder_part(self):
        self._decoder = tensorflow.keras.models.Sequential(name='decoder')
        self._decoder.add(tensorflow.keras.layers.Dense(512, activation=self._decoder_activation,
                                                        input_dim=16 * self._number_of_classes))
        self._decoder.add(tensorflow.keras.layers.Dense(1024, activation=self._decoder_activation))
        self._decoder.add(tensorflow.keras.layers.Dense(numpy.prod(self._input_shape), activation='sigmoid'))
        self._decoder.add(tensorflow.keras.layers.Reshape(target_shape=self._input_shape, name='out_put'))

    def get_training_and_evaluation_model(self):
        self._input = tensorflow.keras.layers.Input(shape=self._input_shape, batch_size=self._batch_size)
        self._capsule_part()
        labels = tensorflow.keras.layers.Input(shape=(self._number_of_classes,))
        mask_with_labels = self._mask_layer([self._capsule_model, labels])
        mask_without_labels = self._mask_layer(self._capsule_model)
        self._decoder_part()
        training_model = tensorflow.keras.models.Model([self._input, labels],
                                                       [self._capsule_model, self._decoder(mask_with_labels)])
        evaluation_model = tensorflow.keras.models.Model([self._input],
                                                         [self._capsule_model, self._decoder(mask_without_labels)])
        return training_model, evaluation_model
