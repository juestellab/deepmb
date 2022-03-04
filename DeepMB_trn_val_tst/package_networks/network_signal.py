import torch.nn as nn
import torch.nn.functional as F


class NetworkSignal(nn.Module):

  def __init__(self, input_nc, output_nc, kernel_size, padding, use_bias):
    super(NetworkSignal, self).__init__()
    self.conv1 = nn.Conv2d(
      in_channels = input_nc,
      out_channels = output_nc,
      kernel_size = kernel_size,
      padding = padding,
      bias = use_bias,
      padding_mode = 'replicate')

  def forward(self, x):
    x = self.conv1(x)
    return x
