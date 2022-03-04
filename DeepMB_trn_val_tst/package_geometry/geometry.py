import numpy as np
from package_tof.precalculate_transducer_pixel_distances import precalculate_transducer_pixel_distances


class Geometry:
  
  def __init__(
    self,
    SCANNER_ID,
    NB_ELEMENTS,
    F_SAMPLE,
    NB_TIME_SAMPLES,
    DAQ_DELAY,
    CROPPED_SAMPLES,
    ARRAY_RADIUS,
    ARRAY_ANGULAR_COVERAGE,
    NB_PIXELS_X,
    NB_PIXELS_Y,
    IMA_SIZE_X,
    IMA_SIZE_Y):

    ### Scanner
    # Scanner codename
    self.SCANNER_ID = SCANNER_ID
    # Number of transducer piezo elements
    self.NB_ELEMENTS = NB_ELEMENTS
    # Sampling frequency (Hz)
    self.F_SAMPLE = F_SAMPLE
    # Length of each time signal (in signal samples)
    self.NB_TIME_SAMPLES = NB_TIME_SAMPLES
    # DAQ temporal delay between laser excitation and ultrasound recording (in signal samples)
    self.DAQ_DELAY = DAQ_DELAY
    # Number of samples that are cropped from the beginning of the sinogram (in signal samples)
    self.CROPPED_SAMPLES = CROPPED_SAMPLES
    # Total number of signals samples that are removed before processing (in signal samples)
    self.SAMPLES_OFFSET = self.DAQ_DELAY + self.CROPPED_SAMPLES
    # Radius of the detector array (m)
    self.ARRAY_RADIUS = ARRAY_RADIUS
    # Angular coverage of the detector array (degrees)
    self.ARRAY_ANGULAR_COVERAGE = ARRAY_ANGULAR_COVERAGE
    # Angular offset of the detector array to correctly orient the field of view (degrees)
    self.ARRAY_ANGULAR_OFFSET = (180.0 - ARRAY_ANGULAR_COVERAGE) / 2 # TODO ---> THIS MIGHT BE DIFFERENT FOR INVISION
    # Angular coordinates of the transducer elements (degrees)
    self.ARRAY_ANGULAR_STEPS = \
      self.ARRAY_ANGULAR_OFFSET + np.arange(0, NB_ELEMENTS) * (ARRAY_ANGULAR_COVERAGE / (NB_ELEMENTS - 1))
    # Horizontal coordinates of the transducer elements (m)
    self.ELEMENTS_X = ARRAY_RADIUS * np.cos(self.ARRAY_ANGULAR_STEPS * np.pi / 180)
    # Vertical coordinates of the transducer elements (m)
    self.ELEMENTS_Y = ARRAY_RADIUS * np.sin(self.ARRAY_ANGULAR_STEPS * np.pi / 180)

    ### Field of view
    # Horizontal size of the image (px)
    self.NB_PIXELS_X = NB_PIXELS_X
    # Vertical size of the image (px)
    self.NB_PIXELS_Y = NB_PIXELS_Y
    # Horizontal size of the image (m)
    self.IMA_SIZE_X = IMA_SIZE_X
    # Vertical size of the image (m)
    self.IMA_SIZE_Y = IMA_SIZE_Y

    ### Distances, in preparation for time-of-flight computation
    # Pre-calculate the distance from each pixel of the field of view to each element of the array (m)
    self.transducer_pixel_distances = precalculate_transducer_pixel_distances(
      self.ELEMENTS_X, self.ELEMENTS_Y, self.IMA_SIZE_X, self.IMA_SIZE_Y, self.NB_PIXELS_X, self.NB_PIXELS_Y,
      self.NB_ELEMENTS)
