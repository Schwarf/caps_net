from pure_interface import Interface, abstractmethod


class ITorchModule(Interface):
    @abstractmethod
    def forward(self, input_data):
        pass
