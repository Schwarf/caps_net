import tensorflow
from pure_interface import InterfaceError

from tf_implementation.interfaces.i_hyper_parameters import IHyperParameters
from tf_implementation.interfaces.i_network_trainer import INetworkTrainer


class CapsuleNetworkTrainer(INetworkTrainer, object):

    def __init__(self, hyper_parameters: IHyperParameters):
        if not IHyperParameters.provided_by(hyper_parameters):
            raise InterfaceError("Hyper parameters object does not derive from IHyperParameters interface.")
        self._model = None
        self._training_input = None
        self._training_labels = None
        self._losses = None
        self._loss_weights = None
        self._optimizer = None
        self._metrics = None
        self._hyper_parameters = hyper_parameters

    def set_losses(self, losses, loss_weights=None):
        self._losses = losses
        if loss_weights is None:
            self._loss_weights = [1.0] * len(losses)

    def set_optimizer(self, optimizer):
        self._optimizer = optimizer

    def set_training_data(self, training_input, training_labels):
        pass

    def train(self):
        self._model.compile(
            optimizer=self._optimizer(learning_rate=self._hyper_parameters.learning_rate),
            loss=self._losses,
            loss_weights=self._loss_weights,
            metrics=self._metrics)

    def set_model(self, model):
        self._model = model

    def set_data(self, training_input, training_labels):
        self._training_input = training_input
        self._training_labels = training_labels

    def set_metrics(self, metrics):
        pass
