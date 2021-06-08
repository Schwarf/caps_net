from pure_interface import Interface, abstractmethod


class ICapsuleModel(Interface):
    @abstractmethod
    def get_training_and_evaluation_model(self):
        pass
