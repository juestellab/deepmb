import torch.nn as nn


class EncodeSoS(nn.Module):

  def __init__(self, sos_range, sos_encoding):
    super(EncodeSoS, self).__init__()
    self.sos_encoding = sos_encoding
    self.sos_min = min(sos_range)
    self.sos_max = max(sos_range)


  def forward(self, sos):

    # SoS values are used as is (e.g., [1492, 1520])
    if self.sos_encoding == 'CHANNEL_AS_IS':
      sos_encoded = sos

    # SoS values are shifted to [0, 1], values are rather unevenly spread (e.g., [0.9836..., 1.0])
    elif self.sos_encoding == 'CHANNEL_HARD_CAP':
      sos_encoded = sos / self.sos_max

    # SoS values are shifted to [1, 2], values are rather unevenly spread (e.g., [1.9836..., 2.0])
    elif self.sos_encoding == 'CHANNEL_HARD_CAP_PLUS':
      sos_encoded = (sos / self.sos_max) + 1

    # SoS values are shifted to [0, 1] more evenly than "HARD_CAP"
    elif self.sos_encoding == 'CHANNEL_SOFT_CAP':
      sos_encoded = (sos - self.sos_min) / (self.sos_max - self.sos_min)

    # SoS values are shifted to [1, 2] more evenly that "HARD_CAP_PLUS"
    elif self.sos_encoding == 'CHANNEL_SOFT_CAP_PLUS':
      sos_encoded = (sos - self.sos_min) / (self.sos_max - self.sos_min) + 1

    # SoS values are shifted to consecutive integer values [0, 1, 2, ...]
    # With two SoS values, this is equivalent to "SOFT_CAP"
    elif self.sos_encoding == 'CHANNEL_CATEGORICAL':
      # Below is an example with only two values (cases must be specified manually for each different dataset)
      if sos == self.sos_min:
        sos_encoded = 0
      else:
        sos_encoded = 1

    # SoS values are shifted to consecutive strictly positive integer values [1, 2, 3, ...]
    # With two SoS values, this is equivalent to "SOFT_CAP"
    # Cases must be specified manually for each different dataset
    elif self.sos_encoding == 'CHANNEL_CATEGORICAL_PLUS':
      # Below is an example with only two values (cases must be specified manually for each different dataset)
      if sos == self.sos_min:
        sos_encoded = 1
      else:
        sos_encoded = 2

    return sos_encoded
