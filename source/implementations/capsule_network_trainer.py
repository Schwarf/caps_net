from interfaces.i_network_trainer import INetworkTrainer


class CapsuleNetworkTrainer(INetworkTrainer, object):
    def __init__(self):
        self._model = None
        self._data = None


    def train(self):
        pass

    def set_model(self, model):
        self._model = model

    def set_data(self, training_data):
        self._data = training_data