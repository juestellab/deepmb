import torch


class ToF():


  def __init__(self, transducer_pixel_distances, scanner_sampling_frequency, samples_offset):
    self.transducer_pixel_distances = transducer_pixel_distances.float()
    self.scanner_sampling_frequency = scanner_sampling_frequency
    self.samples_offset = samples_offset


  def compute_tof(self, speed_of_sounds):

    # Reshape data (batch size is first dimension)
    transducer_pixel_distances_expanded = self.transducer_pixel_distances.expand(len(speed_of_sounds), -1, -1, -1)
    speed_of_sounds = speed_of_sounds.reshape(len(speed_of_sounds), 1, 1, 1)

    # Calculate "tof", namely the time-of-flight values in m/s
    tof = torch.div(transducer_pixel_distances_expanded, speed_of_sounds)

    # Set "tof" in "signal samples" units and subtract the samples offset
    tof = tof * self.scanner_sampling_frequency - self.samples_offset

    return tof


  def cast_tof_for_linear_itp(self, tof):

    # Prepare indices and coefficients for linear interpolation
    tof_floor        = torch.floor(tof)       # Round down
    tof_ceil         = torch.ceil(tof)        # Round up
    tof_floor_factor = 1 - (tof - tof_floor)  # Determine the proportion of the "down" component
    tof_ceil_factor  = 1 - tof_floor_factor   # Determine the proportion of the "up" component

    # Cast to proper type
    tof_floor        = tof_floor.long()
    tof_ceil         = tof_ceil.long()
    tof_floor_factor = tof_floor_factor.float()
    tof_ceil_factor  = tof_ceil_factor.float()

    return tof_floor, tof_ceil, tof_floor_factor, tof_ceil_factor
