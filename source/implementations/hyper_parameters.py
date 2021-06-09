from pure_interface import InterfaceError

from source.interfaces.i_hyper_parameters import IHyperParameters
from source.interfaces.i_hyper_parameter_builder import IHyperParameterBuilder


class HyperParameters(IHyperParameters, object):
    def __init__(self, builder: IHyperParameterBuilder):
        if not IHyperParameterBuilder.provided_by(builder):
            raise InterfaceError(f"Builder object does not inherit from IHyperParameterBuilder")
        self._number_of_epochs = builder.number_of_epochs
        self._batch_size = builder.batch_size
        self._learning_rate = builder.learning_rate
        self._learning_rate_decay_factor = builder.learning_rate_decay_factor
        self._scale_factor_for_reconstruction_loss = builder.scale_factor_for_reconstruction_loss
        self._number_of_routings = builder.number_of_routings

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

    @property
    def number_of_routings(self):
        return self._number_of_routings
