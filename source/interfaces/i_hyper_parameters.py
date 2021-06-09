from pure_interface import Interface, abstractmethod


class IHyperParameters(Interface):
    @property
    @abstractmethod
    def number_of_epochs(self):
        pass

    @property
    @abstractmethod
    def batch_size(self):
        pass

    @property
    @abstractmethod
    def learning_rate(self):
        pass

    @property
    @abstractmethod
    def learning_rate_decay_factor(self):
        pass

    @property
    @abstractmethod
    def scale_factor_for_reconstruction_loss(self):
        pass

    @property
    @abstractmethod
    def number_of_routings(self):
        pass
