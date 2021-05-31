from pure_interface import Interface, abstractmethod


class ICapsuleModel(Interface):
    @abstractmethod
    def model(self):
        pass
