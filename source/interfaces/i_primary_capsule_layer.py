from pure_interface import Interface, abstractmethod


class IPrimaryCapsuleLayer(Interface):
    def apply(self, input_features):
        pass
