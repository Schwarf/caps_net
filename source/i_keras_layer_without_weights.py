from pure_interface import Interface, abstractmethod


class IKerasLayerWithoutWeights(Interface):
    @abstractmethod
    def call(self, inputs, training):
        pass

    @abstractmethod
    def compute_output_shape(self, input_shape):
        pass

    @abstractmethod
    def get_config(self):
        pass
