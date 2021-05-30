import tensorflow
from squash_activation import SquashActivation


class PrimaryCapsuleLayer():
    def __init__(self, dimension_of_capsule, number_of_channels, kernel_size, strides, padding):
        if dimension_of_capsule is None or dimension_of_capsule < 1:
            raise ValueError(f"The dimension of the capsule must be larger than zero but is '{dimension_of_capsule}'.")
        if number_of_channels is None or number_of_channels < 1:
            raise ValueError(f"The number of channels must be larger than zero but is '{number_of_channels}'.")
        if kernel_size is None or kernel_size < 1:
            raise ValueError(f"The kernel size must be larger than zero but is '{number_of_channels}'.")
        if strides is None or strides < 1:
            raise ValueError(f"The strides must be larger than zero but is '{number_of_channels}'.")
        self._number_of_filters = number_of_channels * dimension_of_capsule
        self._dimension_of_capsule = dimension_of_capsule
        self._kernel_size = kernel_size
        self._strides = strides
        self._padding = padding
        self._squash_activation = SquashActivation()

    def apply(self, input):

        output = tensorflow.keras.layers.Conv2D(filters=self._number_of_filters, kernel_size=self._kernel_size,
                                                strides=self._strides, padding=self._padding,
                                                name='primary_capsule_layer_conv2d')(input)
        reshaped_output = tensorflow.keras.layers.Reshape(target_shape=[-1, self._dimension_of_capsule],
                                                          name='primary_capsule_layer_reshape')(output)
        return tensorflow.keras.layers.Lambda(self._squash_activation.apply, name='primarycap_squash')(reshaped_output)
