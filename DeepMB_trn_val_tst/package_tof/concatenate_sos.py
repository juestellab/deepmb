import torch
import torch.nn as nn
from .encode_sos import EncodeSoS


# ----------------------------------------------------------------
class ConcatenateSos(nn.Module):

  def __init__(self, sos_range, sos_encoding, sos_normalization_to_input_data_range, device):
    super(ConcatenateSos, self).__init__()

    # One-hot encoding, where the non-zero code value is a pre-determined factor
    if sos_encoding == 'ONEHOT_STATIC':
      self.apply_sos_concatenation = ConcatenateSosAsOneHot(sos_range, sos_normalization_to_input_data_range, device)

    # One-hot encoding, where the non-zero code value is variable and signal-specific
    elif sos_encoding == 'ONEHOT_DYNAMIC':
      self.apply_sos_concatenation = ConcatenateSosAsOneHotWithSignalSpecificIntensity(sos_range, device)

    else:
      # Do not one-hot encode the SoS; instead encode the SoS value as a single channel
      self.apply_sos_concatenation = \
        ConcatenateSosAsChannel(sos_range, sos_encoding, sos_normalization_to_input_data_range)

  def forward(self, x, sos):
    return self.apply_sos_concatenation(x, sos)


# ----------------------------------------------------------------
class ConcatenateSosAsOneHotWithSignalSpecificIntensity(nn.Module):

  def __init__(self, sos_range, device):
    super(ConcatenateSosAsOneHotWithSignalSpecificIntensity, self).__init__()
    self.sos_range = torch.tensor(sos_range, device=device, dtype=torch.float)
    self.nb_sos = len(sos_range)
    self.device = device

  def forward(self, x, sos):
    batch_size, num_channels, dim_y, dim_x = x.shape
    encoded_sos_values = torch.zeros(batch_size, self.nb_sos, device=self.device, dtype=torch.float)
    for i_batch in range(batch_size):
      # Calculate the median value of the absolute signal intensity, then reformat and reshape accordingly
      median_abs_signal_intensity = \
        torch.median(torch.abs(x[i_batch,:,:,:].view(-1).type(torch.float))).expand(self.nb_sos)
      # Encode
      encoded_sos_values[i_batch] = torch.mul(median_abs_signal_intensity, (self.sos_range == sos[i_batch]))
    encoded_sos_values = encoded_sos_values.view(batch_size, self.nb_sos, 1, 1).expand(-1, -1, dim_y, dim_x)
    return torch.cat([x, encoded_sos_values], dim=1)


# ----------------------------------------------------------------
class ConcatenateSosAsOneHot(nn.Module):

  def __init__(self, sos_range, sos_normalization_to_input_data_range, device):
    super(ConcatenateSosAsOneHot, self).__init__()
    self.sos_range = torch.tensor(sos_range, device=device, dtype=torch.float)
    self.nb_sos = len(sos_range)
    self.device = device
    self.sos_normalization_to_input_data_range = \
      torch.tensor(sos_normalization_to_input_data_range, device=device, dtype=torch.float).expand(self.nb_sos)

  def forward(self, x, sos):
    batch_size, num_channels, dim_y, dim_x = x.shape
    encoded_sos_values = torch.zeros(batch_size, self.nb_sos, device=self.device, dtype=torch.float)
    for i_batch in range(batch_size):
      encoded_sos_values[i_batch] = torch.mul(
        self.sos_normalization_to_input_data_range, (self.sos_range == sos[i_batch]))
    encoded_sos_values = encoded_sos_values.view(batch_size, self.nb_sos, 1, 1).expand(-1, -1, dim_y, dim_x)
    return torch.cat([x, encoded_sos_values], dim=1)


# ----------------------------------------------------------------
class ConcatenateSosAsChannel(nn.Module):

  def __init__(self, sos_range, sos_encoding, sos_normalization_to_input_data_range):
    super(ConcatenateSosAsChannel, self).__init__()
    self.encode_sos = EncodeSoS(sos_range, sos_encoding)
    self.sos_normalization_to_input_data_range = sos_normalization_to_input_data_range

  def forward(self, x, sos):
    batch_size, num_channels, dim_y, dim_x = x.shape
    sos_encoded_and_normalized = self.encode_sos(sos) * self.sos_normalization_to_input_data_range
    return torch.cat([x, sos_encoded_and_normalized.reshape(batch_size, 1, 1, 1).expand(-1, 1, dim_y, dim_x)], dim=1)
