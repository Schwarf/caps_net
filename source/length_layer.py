from i_layer_without_weights import IKerasLayerWithoutWeights
import tensorflow


class LengthLayer(IKerasLayerWithoutWeights, tensorflow.keras.layers.Layer, object):
    def __init__(self):
        super(LengthLayer, self).__init__()

    def call(self, input, **kwargs):
        return tensorflow.sqrt(tensorflow.reduce_sum(tensorflow.square(input), -1) +
                               tensorflow.keras.backend.epsilon())

    def compute_output_shape(self, input_shape):
        return input_shape[:-1]

    def get_config(self):
        config = super(LengthLayer, self).get_config()
        return config
