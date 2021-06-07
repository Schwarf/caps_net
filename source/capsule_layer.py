import tensorflow

from i_layer_with_weights import IKerasLayerWithWeights
from squash_activation import SquashActivation


class KerasLayerWithWeights(IKerasLayerWithWeights, tensorflow.keras.layers.Layer, object):
    def __init__(self, number_of_capsules, dimension_of_capsule, number_of_routings):
        super(KerasLayerWithWeights, self).__init__()
        if number_of_capsules < 1:
            raise ValueError(f"The number of capsules must be larger than 0 but is {number_of_capsules}")
        if dimension_of_capsule < 1:
            raise ValueError(f"The dimension of a capsule must be larger than 0 but is {dimension_of_capsule}")
        if number_of_routings < 1:
            raise ValueError(f"The number_of_routings must be larger than 0 but is {number_of_routings}")
        self._number_of_routings = number_of_routings
        self._number_of_capsules = number_of_capsules
        self._dimension_of_capsule = dimension_of_capsule
        self._kernel_initializer = tensorflow.keras.initializers.get('glorot_uniform')
        self._number_of_input_capsules = None
        self._dimension_of_input_capsules = None
        self._weight_matrix = None
        self._squash_activation = SquashActivation()

    def build(self, input_shape):
        assert len(input_shape) >= 3
        self._number_of_input_capsules = input_shape[1]
        self._dimension_of_input_capsules = input_shape[2]
        weight_matrix_shape = [self._number_of_capsules, self._number_of_input_capsules, self._dimension_of_capsule,
                               self._dimension_of_input_capsules]
        self._weight_matrix = self.add_weight(shape=weight_matrix_shape, initializer=self._kernel_initializer,
                                              name='weight_matrix', trainable=True)
        self.built = True

    def call(self, inputs, training=None):
        batch_size = inputs.shape[0]
        # Reshape inputs from [None, number_of_input_capsules, dimension_of_input_capsules]
        # to [None, 1, number_of_input_capsules, dimension_of_input_capsules, 1]
        expanded_inputs = tensorflow.expand_dims(tensorflow.expand_dims(inputs, 1), -1)

        # Copy the input matrix along the 2nd dimension number_of_capsules x times to simplify the
        # multiplication with the weight_matrix.
        # tiled_inputs shape is [None, number_of_capsules, number_of_input_capsules, dimension_of_input_capsules, 1]
        tiled_inputs = tensorflow.tile(expanded_inputs, [1, self._number_of_capsules, 1, 1, 1])

        # check dimension again
        inputs_times_weight_matrix = tensorflow.squeeze(
            tensorflow.map_fn(lambda x: tensorflow.matmul(self.weight_matrix, x), elems=tiled_inputs))

        # Initialize the coupling coefficients
        coupling_coefficients = tensorflow.zeros(
            shape=[batch_size, self._number_of_capsules, 1, self._number_of_input_capsules])

        output_vectors = None
        for round in range(self._number_of_routings):
            softmax_activation_primary_capsule = tensorflow.nn.softmax(coupling_coefficients, axis=1)
            output_vectors = self._squash_activation.apply(
                tensorflow.matmul(softmax_activation_primary_capsule, inputs_times_weight_matrix))  # [None, 10, 1, 16]
            if round < self._number_of_routings - 1:
                # update couplings
                coupling_coefficients += tensorflow.matmul(output_vectors, inputs_times_weight_matrix, transpose_b=True)

        return tensorflow.squeeze(output_vectors)

    def compute_output_shape(self, input_shape):
        return tuple([None, self._number_of_capsules, self._dimension_of_capsule])

    def get_config(self):
        config = {
            'number_of_capsules': self._number_of_capsules,
            'dimension_of_capsule': self._dimension_of_capsule,
            'number_of_routings': self._number_of_routings
        }
        base_config = super(KerasLayerWithWeights, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
