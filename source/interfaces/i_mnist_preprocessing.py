from pure_interface import abstractmethod, Interface


class IMNISTPreprocessing(Interface):
    @abstractmethod
    def apply(self, mnist_data):
        pass
