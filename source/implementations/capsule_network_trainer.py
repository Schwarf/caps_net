import tensorflow
from interfaces.i_network_trainer import INetworkTrainer


def margin_loss(true_label, predicted_label):
    loss = true_label * tensorflow.square(tensorflow.maximum(0., 0.9 - predicted_label)) + \
           0.5 * (1 - true_label) * tensorflow.square(tensorflow.maximum(0., predicted_label - 0.1))

    return tensorflow.reduce_mean(tensorflow.reduce_sum(loss, 1))


class CapsuleNetworkTrainer(INetworkTrainer, object):
    def __init__(self):
        self._model = None
        self._input = None
        self._labels = None

    def train(self):
        self._model.compile(optimizer=tensorflow.keras.optimizers.Adam(lr=args.lr),
                      loss=[margin_loss, 'mse'],
                      loss_weights=[1., args.lam_recon],
                      metrics={'capsnet': 'accuracy'})

    def set_model(self, model):
        self._model = model

    def set_data(self, training_input, training_labels):
        self._input = training_input
        self._labels = training_labels