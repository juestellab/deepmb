import torch.nn as nn


class NetworkDamping(nn.Module):

  # (N, C, H, W)_input = (1, CH_IN=256, 384, 384)
  # (N, C, H, W)_output = (1, CH_OUT, 384, 384)
  # The purpose of this network is to gradually adapt (damp) the number of channels...
  # ...from "CH_IN=256" (1 per element) down to "CH_OUT" (number of channels in the outer layer of the subsequent UNet)

  def __init__(self, ch_in, ch_steps, kernel, padding, use_bias):
    super(NetworkDamping, self).__init__()

    self.model = [
      nn.Conv2d(in_channels=ch_in, out_channels=ch_steps[0], kernel_size=kernel, padding=padding, bias=use_bias)]

    for i in range(1, len(ch_steps)):
      self.model += [nn.Conv2d(
        in_channels=ch_steps[i-1], out_channels=ch_steps[i], kernel_size=kernel, padding=padding, bias=use_bias)]

    self.model = nn.Sequential(*self.model)


  def forward(self, x):
    return self.model(x)
