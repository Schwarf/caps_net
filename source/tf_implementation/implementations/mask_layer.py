import tensorflow

from tf_implementation.interfaces.i_layer_without_weights import IKerasLayerWithoutWeights


class MaskLayer(IKerasLayerWithoutWeights, tensorflow.keras.layers.Layer, object):
    def call(self, input_features, **kwargs):
        # if input_features is a list of length 2, than labels are provided and we are in training mode.
        if isinstance(input_features, list):
            if len(input_features) == 2:
                input_features, mask = input_features
            else:
                raise ValueError(f"Length of list in MaskLayer is {len(input_features)} and not 2.")
        # if input_features is a single variable we are in evaluation mode and use the capsule with maximal length
        # (=probability)
        else:
            length = tensorflow.sqrt(tensorflow.reduce_sum(tensorflow.square(input_features), -1))
            mask = tensorflow.one_hot(indices=tensorflow.argmax(length, 1), depth=length.shape[1])

        masked = tensorflow.keras.backend.batch_flatten(input_features * tensorflow.expand_dims(mask, -1))
        return masked

    def compute_output_shape(self, input_features_shape):
        if type(input_features_shape[0]) is tuple:  # true label provided
            return tuple([None, input_features_shape[0][1] * input_features_shape[0][2]])
        else:  # no true label provided
            return tuple([None, input_features_shape[1] * input_features_shape[2]])

    def get_config(self):
        config = super(MaskLayer, self).get_config()
        return config
