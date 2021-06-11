from pure_interface import Interface, abstractmethod

from tf_implementation.interfaces.i_hyper_parameters import IHyperParameters


class IHyperParameterBuilder(IHyperParameters, Interface):
    @abstractmethod
    def set_number_of_epochs(self, number_of_epochs):
        pass

    @abstractmethod
    def set_batch_size(self, batch_size):
        pass

    @abstractmethod
    def set_learning_rate(self, learning_rate):
        pass

    @abstractmethod
    def set_learning_rate_decay_factor(self, learning_rate_decay_factor):
        pass

    @abstractmethod
    def set_scale_factor_for_reconstruction_loss(self, scale_factor_for_reconstruction_loss):
        pass

    @abstractmethod
    def set_number_of_routings(self, number_of_routings):
        pass
