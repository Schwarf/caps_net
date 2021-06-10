from pure_interface import Interface, abstractmethod


class IMNISTData(Interface):
    @property
    @abstractmethod
    def training_input(self):
        pass

    @training_input.setter
    @abstractmethod
    def training_input(self, training_input):
        pass

    @property
    @abstractmethod
    def training_labels(self):
        pass

    @training_labels.setter
    @abstractmethod
    def training_labels(self, training_labels):
        pass

    @property
    @abstractmethod
    def test_input(self):
        pass

    @test_input.setter
    @abstractmethod
    def test_input(self, test_input):
        pass

    @property
    @abstractmethod
    def test_labels(self):
        pass

    @test_labels.setter
    @abstractmethod
    def test_labels(self, test_labels):
        pass
