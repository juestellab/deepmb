import torch
import torch.nn as nn
from package_tof.tof import ToF


class DelayNoSumGather(nn.Module):


  def __init__(self, transducer_pixel_distances, scanner_sampling_frequency, samples_offset):
    super(DelayNoSumGather, self).__init__()

    # Apparatus to compute the ToF given a specified SoS
    self.tof = ToF(transducer_pixel_distances, scanner_sampling_frequency, samples_offset)
    self.nb_pixels_y, self.nb_pixels_x, self.nb_elements = self.tof.transducer_pixel_distances.shape


  def forward(self, input_sinogram, sos):

    batch_size = input_sinogram.shape[0]

    # Compute the ToF values necessary for linear interpolation
    tof_floor, tof_ceil, tof_floor_factor, tof_ceil_factor = self.tof.cast_tof_for_linear_itp(self.tof.compute_tof(sos))

    # Reshape and initialize
    tof_floor = tof_floor.view(batch_size, self.nb_pixels_y * self.nb_pixels_x, self.nb_elements)
    tof_ceil = tof_ceil.view(batch_size, self.nb_pixels_y * self.nb_pixels_x, self.nb_elements)
    tof_floor_factor = tof_floor_factor.view(batch_size, self.nb_pixels_y * self.nb_pixels_x, self.nb_elements)
    tof_ceil_factor = tof_ceil_factor.view(batch_size, self.nb_pixels_y * self.nb_pixels_x, self.nb_elements)

    # Remove singleton channel dimension
    input_sinogram = input_sinogram.reshape(batch_size, -1, self.nb_elements)

    # Linear interpolation
    ima = (
      tof_floor_factor * torch.gather(input=input_sinogram, dim=1, index=tof_floor, out=None, sparse_grad=False) +
      tof_ceil_factor * torch.gather(input=input_sinogram, dim=1, index=tof_ceil, out=None, sparse_grad=False))

    # Reshape to image-like output with transducers as channels
    ima = ima.view((
      batch_size, self.nb_pixels_y, self.nb_pixels_x, self.nb_elements)).permute(0, 3, 1, 2).contiguous().float()

    return ima
