from pure_interface import Interface, abstractmethod


class INetworkTrainer(Interface):
    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def set_model(self, model):
        pass

    @abstractmethod
    def set_data(self, training_data):
        pass