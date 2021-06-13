from torch_implementation.interfaces.i_torch_module import ITorchModule
import torch


class ConvolutionLayer(ITorchModule, torch.nn.Module, object):
    def __init__(self, number_of_input_channels, number_of_output_channels, kernel_size, stride_size):
        super(ConvolutionLayer, self).__init__()
        if number_of_input_channels is None or number_of_input_channels < 1:
            raise ValueError(
                f"The number of input channels must be larger than zero but is '{number_of_input_channels}'.")
        if number_of_output_channels is None or number_of_output_channels < 1:
            raise ValueError(
                f"The number of output channels must be larger than zero but is '{number_of_output_channels}'.")
        if kernel_size is None or kernel_size < 1:
            raise ValueError(f"The kernel size must be larger than zero but is '{kernel_size}'.")
        if stride_size is None or stride_size < 1:
            raise ValueError(f"The stride size must be larger than zero but is '{stride_size}'.")

        self._convolutional_layer = torch.nn.Conv2d(in_channels=number_of_input_channels,
                                                    out_channels=number_of_output_channels,
                                                    kernel_size=kernel_size,
                                                    stride=stride_size)


    def forward(self, input_data):
        return torch.nn.ReLU(self._convolutional_layer(input_data))
