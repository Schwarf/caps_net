import tensorflow

from interfaces.i_layer_without_weights import IKerasLayerWithoutWeights


class MaskLayer(IKerasLayerWithoutWeights, tensorflow.keras.layers.Layer, object):
    def __init__(self):
        super(MaskLayer, self).__init__()

    def call(self, input, **kwargs):
        # if input is a list of length 2, than labels are provided and we are in training mode.
        if isinstance(input, list):
            if len(input) == 2:
                input, mask = input
            else:
                raise ValueError(f"Length of list in MaskLayer is {len(input)} and not 2.")
        # if input is a single variable we are in evaluation mode and use the capsule with maximal length (=probability)
        else:
            length = tensorflow.sqrt(tensorflow.reduce_sum(tensorflow.square(input), -1))
            mask = tensorflow.one_hot(indices=tensorflow.argmax(length, 1), depth=length.shape[1])

        masked = tensorflow.keras.backend.batch_flatten(input * tensorflow.expand_dims(mask, -1))
        return masked


    def compute_output_shape(self, input_shape):
        return input_shape[:-1]

    def get_config(self):
        config = super(MaskLayer, self).get_config()
        return config
