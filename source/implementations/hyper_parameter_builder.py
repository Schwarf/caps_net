from interfaces.i_hyper_parameter_builder import IHyperParameterBuilder


class HyperParameterBuilder(IHyperParameterBuilder, object):
    def __init__(self):
        self._number_of_epochs = None
        self._batch_size = None
        self._learning_rate = None
        self._learning_rate_decay_factor = None
        self._scale_factor_for_reconstruction_loss = None
        
    def set_number_of_epochs(self, number_of_epochs):
        if number_of_epochs is None:
            raise  ValueError("Number of epochs is 'None'!")
        if number_of_epochs < 1:
            raise ValueError(f"Number of epochs is smaller than 1: {number_of_epochs}")
        self._number_of_epochs = number_of_epochs

    def set_batch_size(self, batch_size):
        if batch_size is None:
            raise  ValueError("Batch size is 'None'!")
        if batch_size < 1:
            raise ValueError(f"Batch size is smaller than 1: {batch_size}")
        self._batch_size = batch_size

    def set_learning_rate(self, learning_rate):
        if learning_rate is None:
            raise  ValueError("Learning rate is 'None'!")
        if learning_rate < 0.0 or learning_rate > 1.0:
            raise ValueError(f"Invalid value for learning rate: {learning_rate}")
        self._learning_rate = learning_rate

    def set_learning_rate_decay_factor(self, learning_rate_decay_factor):
        if learning_rate_decay_factor is None:
            raise  ValueError("Learning rate is 'None'!")
        if learning_rate_decay_factor < 0.0 or learning_rate_decay_factor > 1.0:
            raise ValueError(f"Invalid vaalue for learning-rate-decay-factor: {learning_rate_decay_factor}")
        self._learning_rate_decay_factor = learning_rate_decay_factor

    def set_scale_factor_for_reconstruction_loss(self, scale_factor_for_reconstruction_loss):
        if scale_factor_for_reconstruction_loss is None:
            raise  ValueError("Learning rate is 'None'!")
        if scale_factor_for_reconstruction_loss < 0.0 or scale_factor_for_reconstruction_loss > 1.0:
            raise ValueError(f"Invalid vaalue for learning-rate-decay-factor: {scale_factor_for_reconstruction_loss}")
        self._scale_factor_for_reconstruction_loss = scale_factor_for_reconstruction_loss


    @property
    def number_of_epochs(self):
        return self._number_of_epochs

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def learning_rate(self):
        return self._learning_rate

    @property
    def learning_rate_decay_factor(self):
        return self._learning_rate_decay_factor

    @property
    def scale_factor_for_reconstruction_loss(self):
        return self._scale_factor_for_reconstruction_loss