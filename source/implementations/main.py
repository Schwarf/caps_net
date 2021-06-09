from pure_interface import adapt_args, Interface, abstractmethod
from implementations.capsule_network_trainer import CapsuleNetworkTrainer
from implementations.hyper_parameter_builder import HyperParameterBuilder
from implementations.hyper_parameters import HyperParameters
from interfaces.i_hyper_parameter_builder import IHyperParameterBuilder

class IData(Interface):
    @abstractmethod
    def get(self):
        pass

class Data(IData, object):
    def get(self):
        return 12

class IUser(Interface):
    pass

class User(IUser, object):
    @adapt_args
    def __init__(self, data: IData):
        self._data = data


learning_rate = 0.001
number_of_epochs = 50
batch_size = 50
learning_rate_decay_factor = 0.9
scale_factor_for_reconstruction_loss = 0.392
number_of_routings = 3

parameter_builder = HyperParameterBuilder()
parameter_builder.set_learning_rate(learning_rate)
parameter_builder.set_batch_size(batch_size)
parameter_builder.set_number_of_epochs(number_of_epochs)
parameter_builder.set_scale_factor_for_reconstruction_loss(scale_factor_for_reconstruction_loss)
parameter_builder.set_learning_rate_decay_factor(learning_rate_decay_factor)
print(isinstance(parameter_builder, IHyperParameterBuilder))

hyper_parameters = HyperParameters(parameter_builder)
trainer = CapsuleNetworkTrainer(hyper_parameters)


data = Data()
user = User(data)

x =1