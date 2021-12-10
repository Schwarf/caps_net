import tensorflow


class SquashActivation:
    def __init__(self, squash_axis=-1):
        self._squash_axis = squash_axis

    def apply(self, vector):
        squared_norm = tensorflow.reduce_sum(tensorflow.square(vector), self._squash_axis, keepdims=True)
        scale_factor = squared_norm / (1 + squared_norm) / tensorflow.sqrt(
            squared_norm + tensorflow.keras.backend.epsilon())
        return scale_factor * vector
