from pure_interface import Interface, abstractmethod


class ICapsuleModel(Interface):
    @abstractmethod
    def training_model(self):
        pass

    @abstractmethod
    def evaluation_model(self):
        pass

    @abstractmethod
    def margin_loss(self, true_label, predicted_label):
        pass