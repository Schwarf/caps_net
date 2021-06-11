from pure_interface import Interface, abstractmethod


class INetworkTrainer(Interface):
    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def set_model(self, model):
        pass

    @abstractmethod
    def set_losses(self, losses, loss_weights):
        pass

    @abstractmethod
    def set_optimizer(self, optimizer):
        pass

    @abstractmethod
    def set_training_data(self, training_input, training_labels):
        pass

    @abstractmethod
    def set_validation_data(self, validation_input, validation_labels):
        pass

    @abstractmethod
    def set_metrics(self, metrics):
        pass
