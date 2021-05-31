from pure_interface import Interface, abstractmethod


class ICapsuleModel(Interface):
    @abstractmethod
    def model(self):
        pass

    @abstractmethod
    def length(self, vector):
        pass

