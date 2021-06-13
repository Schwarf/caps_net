import torch

from torch_implementation.interfaces.i_torch_module import ITorchModule


class PrimaryCapsuleLayer(ITorchModule, torch.nn.Module, object):
    def __init__(self, number_of_convolution_input_channels, number_of_convolution_output_channels,
                 kernel_size, stride_size, number_of_primary_capsules):
        super(PrimaryCapsuleLayer, self).__init__()
        if number_of_convolution_input_channels is None or number_of_convolution_input_channels < 1:
            raise ValueError(
                f"The number of input channels must be larger than zero but is "
                f"'{number_of_convolution_input_channels}'.")
        if number_of_convolution_output_channels is None or number_of_convolution_output_channels < 1:
            raise ValueError(
                f"The number of output channels must be larger than zero but is "
                f"'{number_of_convolution_output_channels}'.")
        if kernel_size is None or kernel_size < 1:
            raise ValueError(f"The kernel size must be larger than zero but is '{kernel_size}'.")
        if stride_size is None or stride_size < 1:
            raise ValueError(f"The stride size must be larger than zero but is '{stride_size}'.")
        if number_of_primary_capsules is None or number_of_primary_capsules < 1:
            raise ValueError(f"The number of primary capsules must be larger than zero but is "
                             f"'{number_of_primary_capsules}'.")


    def forward(self, input_data):
        pass
