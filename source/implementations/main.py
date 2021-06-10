import tensorflow

from implementations.capsule_model import CapsuleModel
from implementations.capsule_network_trainer import CapsuleNetworkTrainer
from implementations.hyper_parameter_builder import HyperParameterBuilder
from implementations.hyper_parameters import HyperParameters
from implementations.mnist_data import MNISTData
from interfaces.i_hyper_parameter_builder import IHyperParameterBuilder
from implementations.mnist_preprocessing import MNISTPreprocessing
tensorflow.keras.backend.set_image_data_format('channels_last')
tensorflow.executing_eagerly()

learning_rate = 0.001
number_of_epochs = 50
batch_size = 100
learning_rate_decay_factor = 0.9
scale_factor_for_reconstruction_loss = 0.392
number_of_routings = 3
number_of_classes = 10

parameter_builder = HyperParameterBuilder()
parameter_builder.set_learning_rate(learning_rate)
parameter_builder.set_batch_size(batch_size)
parameter_builder.set_number_of_epochs(number_of_epochs)
parameter_builder.set_scale_factor_for_reconstruction_loss(scale_factor_for_reconstruction_loss)
parameter_builder.set_learning_rate_decay_factor(learning_rate_decay_factor)
parameter_builder.set_number_of_routings(number_of_routings)


hyper_parameters = HyperParameters(parameter_builder)
trainer = CapsuleNetworkTrainer(hyper_parameters)
input_shape = (28, 28, 1)

capsule_model = CapsuleModel(input_shape=input_shape, number_of_classes=number_of_classes,
                             hyper_parameters=hyper_parameters)
training_model, evaluation_model = capsule_model.get_training_and_evaluation_model()
mnist_data = MNISTData()
preprocessor = MNISTPreprocessing()
mnist_data = preprocessor.apply(mnist_data=mnist_data)

trainer.set_model(training_model)
trainer.set_training_data(mnist_data.training_input, mnist_data.training_labels)

trainer.set_losses()


