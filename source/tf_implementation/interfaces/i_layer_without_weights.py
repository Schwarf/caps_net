from pure_interface import Interface, abstractmethod


class IKerasLayerWithoutWeights(Interface):
    @abstractmethod
    def call(self, input_features, **kwargs):
        pass

    @abstractmethod
    def compute_output_shape(self, input_features_shape):
        pass

    @abstractmethod
    def get_config(self):
        pass
