import torch
import torch.nn as nn
from .network_unet import Unet
from .network_signal import NetworkSignal
from .network_damping import NetworkDamping
from .delay_no_sum_gather import DelayNoSumGather
from package_tof.concatenate_sos import ConcatenateSos


# ----------------------------------------------------------------
class Net(nn.Module):
  def __init__(self, transducer_pixel_distances, scanner_sampling_frequency, samples_offset, device, p, g):
    super(Net, self).__init__()

    # Constants
    NUM_CH_NETWORK_IN = 1
    NUM_CH_NETWORK_OUT = 1
    NUM_CH_DELAY_IN = 1
    NUM_CH_DELAY_OUT = g.NB_ELEMENTS

    # SoS concatenation
    self.concatenate_sos = ConcatenateSos(p.SOS_RANGE, p.SOS_ENCODING, p.SOS_NORMALIZATION_TO_INPUT_DATA_RANGE, device)
    self.provide_sos_to_net_sig = p.PROVIDE_SOS_TO_NET_SIG
    self.provide_sos_to_net_ima = p.PROVIDE_SOS_TO_NET_IMA

    # Signal domain
    num_ch_sig_in = calculate_n_channels_with_encoded_sos(
      NUM_CH_NETWORK_IN, p.PROVIDE_SOS_TO_NET_SIG, p.SOS_ENCODING, p.SOS_RANGE)
    num_ch_sig_out = NUM_CH_DELAY_IN
    if p.SIG_NET_ARCHI == 'UNET':
      self.net_sig = Unet(
        input_nc=num_ch_sig_in, output_nc=num_ch_sig_out, n_layers=p.NB_LAYERS_UNET_SIG, ngf=p.NGF_UNET_SIG,
        kernel_size=p.KERNEL_SIZE_UNET_SIG, padding=p.PAD_SIZE_UNET_SIG, use_bias=p.USE_BIAS_SIG)
    elif p.SIG_NET_ARCHI == 'CONV2D':
      self.net_sig = NetworkSignal(
        input_nc=num_ch_sig_in, output_nc=num_ch_sig_out, kernel_size=p.KERNEL_SIZE_UNET_SIG,
        padding=p.PAD_SIZE_UNET_SIG, use_bias=p.USE_BIAS_SIG)
    elif p.SIG_NET_ARCHI == 'NONE':
      self.net_sig = IdentityFunction()
    else:
      raise Exception('Unknown SIG_NET_ARCHI: ' + p.SIG_NET_ARCHI)

    # Domain transfer
    self.net_delay = DelayNoSumGather(transducer_pixel_distances, scanner_sampling_frequency, samples_offset)

    # Image domain
    num_ch_ima_in = calculate_n_channels_with_encoded_sos(
      NUM_CH_DELAY_OUT, p.PROVIDE_SOS_TO_NET_IMA, p.SOS_ENCODING, p.SOS_RANGE)
    num_ch_ima_out = NUM_CH_NETWORK_OUT
    if p.IMA_NET_ARCHI == 'UNET':
      self.net_ima = Unet(
        input_nc=num_ch_ima_in, output_nc=num_ch_ima_out, n_layers=p.NB_LAYERS_UNET_IMA, ngf=p.NGF_UNET_IMA,
        kernel_size=p.KERNEL_SIZE_UNET_IMA, padding=p.PAD_SIZE_UNET_IMA, use_bias=p.USE_BIAS_IMA)
    elif p.IMA_NET_ARCHI == 'DAMPING+UNET':
      damping = NetworkDamping(
        num_ch_ima_in, p.CH_STEPS_DAMPING, p.KERNEL_SIZE_DAMPING, p.PAD_SIZE_DAMPING, use_bias=p.USE_BIAS_IMA)
      unet = Unet(
        input_nc=p.CH_STEPS_DAMPING[-1], output_nc=num_ch_ima_out, n_layers=p.NB_LAYERS_UNET_IMA, ngf=p.NGF_UNET_IMA,
        kernel_size=p.KERNEL_SIZE_UNET_IMA, padding=p.PAD_SIZE_UNET_IMA, use_bias=p.USE_BIAS_IMA)
      self.net_ima = nn.Sequential(damping, unet)
    elif p.IMA_NET_ARCHI == 'SUM+UNET':
      net_sum = SumChannels(g.NB_ELEMENTS)
      unet = Unet(
        input_nc=1, output_nc=num_ch_ima_out, n_layers=p.NB_LAYERS_UNET_IMA, ngf=p.NGF_UNET_IMA,
        kernel_size=p.KERNEL_SIZE_UNET_IMA, padding=p.PAD_SIZE_UNET_IMA, use_bias=p.USE_BIAS_IMA)
      self.net_ima = nn.Sequential(net_sum, unet)
    elif p.IMA_NET_ARCHI == 'SUM':
      self.net_ima = SumChannels(g.NB_ELEMENTS)
    elif p.IMA_NET_ARCHI == 'DAMPING':
      self.net_ima = NetworkDamping(
        num_ch_ima_in, p.CH_STEPS_DAMPING, p.KERNEL_SIZE_DAMPING, p.PAD_SIZE_DAMPING, use_bias=p.USE_BIAS_IMA)
    else:
      raise Exception('Unknown IMA_NET_ARCHI: ' + p.IMA_NET_ARCHI)

    # Final activation
    if p.OUTPUT_ACTIVATION == 'ABS':
      self.net_output_activation = AbsActivation()
    elif p.OUTPUT_ACTIVATION == 'SOFTPLUS':
      self.net_output_activation = nn.Softplus(beta=1, threshold=20)
    else:
      raise Exception('Unknown OUTPUT_ACTIVATION: ' + p.OUTPUT_ACTIVATION)

  # ----------------------------------------------------------------
  def forward(self, x, sos):
    # Signal domain: [BATCH_SIZE, NB_EL(+SOS), DEPTH] ---> [BATCH_SIZE, NB_EL, DEPTH]
    if self.provide_sos_to_net_sig:
      x = self.concatenate_sos(x, sos)
    x = self.net_sig(x)
    # Domain transfer (Delay-No-Sum): [BATCH_SIZE, NB_EL, DEPTH] ---> [BATCH_SIZE, NB_EL, PIXELS_X, PIXELS_Y]
    x = self.net_delay(x, sos)
    # Image domain: [BATCH_SIZE, NB_CHANNELS_UNET(+SOS), PIXELS_X, PIXELS_Y] ---> [BATCH_SIZE, 1, PIXELS_X, PIXELS_Y]
    if self.provide_sos_to_net_ima:
      x = self.concatenate_sos(x, sos)
    x = self.net_ima(x)
    # Non-negativity: [BATCH_SIZE, 1, PIXELS_X, PIXELS_Y] ---> [BATCH_SIZE, 1, PIXELS_X, PIXELS_Y]
    x = self.net_output_activation(x)
    return x


# ----------------------------------------------------------------
class AbsActivation(nn.Module):
  def __init__(self):
    super(AbsActivation, self).__init__()
  def forward(self, x):
    return torch.abs(x)


# ----------------------------------------------------------------
class SumChannels(nn.Module):
  def __init__(self, nb_channels):
    super(SumChannels, self).__init__()
    self.nb_channels = nb_channels
  def forward(self, x):
    return torch.sum(x, dim=1, keepdim=True) / self.nb_channels


# ----------------------------------------------------------------
class IdentityFunction(nn.Module):
  def __init__(self):
    super(IdentityFunction, self).__init__()
  def forward(self, x):
    return x


# ----------------------------------------------------------------
def calculate_n_channels_with_encoded_sos(ch_basis, provide_sos, sos_encoding, sos_range):
  if provide_sos and sos_encoding.startswith('ONEHOT_'):
    ch_additional = len(sos_range)
  elif provide_sos and not sos_encoding.startswith('ONEHOT_'):
    ch_additional = 1
  else:
    ch_additional = 0
  return ch_basis + ch_additional
