from package_geometry import geometry

# Characterize the selected imaging device and the desired field of view
# ---> (cf. associated parameter values in "path_to_rec_toolbox/reconstruction_model_based/models/Probe.m")
# ---> (cf. associated parameter values in "DeepMB/process_data_generation/data_generation_utils/define_msot_model.m")

def define_scanner_and_fov_geometry(SCANNER_ID):


  # ----------------------------------------------------------------
  # This is the configuration that was used for the DeepMB paper
  if SCANNER_ID == 'Acuity_256e_125d':

    ### Scanner
    # Number of transducer piezo elements
    NB_ELEMENTS = 256
    # Sampling frequency (Hz)
    F_SAMPLE = 40000000
    # Length of each time signal (in signal samples, nicely divisible by 2^7)
    NB_TIME_SAMPLES = 1920
    # DAQ temporal delay between laser excitation and ultrasound recording (in signal samples)
    DAQ_DELAY = 8
    # Number of samples that are cropped from the beginning of the sinogram (in signal samples)
    CROPPED_SAMPLES = 110
    # Radius of the detector array (m)
    ARRAY_RADIUS = 0.040
    # Angular coverage of the detector array (degrees)
    ARRAY_ANGULAR_COVERAGE = 125.0

    ### Field of view
    # Horizontal size of the image (px, nicely divisible by 2^7)
    NB_PIXELS_X = 416
    # Vertical size of the image (px, nicely divisible by 2^7)
    NB_PIXELS_Z = 416
    # Horizontal size of the image (m, yields a round pixel size of 100 um)
    IMA_SIZE_X = 0.0415
    # Vertical size of the image (m, yields a round pixel size of 100 um)
    IMA_SIZE_Z = 0.0415


  # ----------------------------------------------------------------
  elif SCANNER_ID == 'inVision_256e_270d':

    ### Scanner
    # Number of transducer piezo elements
    NB_ELEMENTS = 256
    # Sampling frequency (Hz)
    F_SAMPLE = 40000000
    # Length of each time signal (in signal samples)
    NB_TIME_SAMPLES = 1920
    # DAQ temporal delay between laser excitation and ultrasound recording (in signal samples)
    DAQ_DELAY = 0
    # Number of samples that are cropped from the beginning of the sinogram (in signal samples)
    CROPPED_SAMPLES = 110
    # Radius of the detector array (m)
    ARRAY_RADIUS = 0.040
    # Angular coverage of the detector array (degrees)
    ARRAY_ANGULAR_COVERAGE = 270.0

    ### Field of view
    # Horizontal size of the image (px)
    NB_PIXELS_X = 512
    # Vertical size of the image (px)
    NB_PIXELS_Z = 512
    # Horizontal size of the image (m)
    IMA_SIZE_X = 0.02555
    # Vertical size of the image (m)
    IMA_SIZE_Z = 0.02555


  # ----------------------------------------------------------------
  elif SCANNER_ID == 'Your_New_Scanner_Type':

    ### Scanner
    # Number of transducer piezo elements
    NB_ELEMENTS = '???'
    # Sampling frequency (Hz)
    F_SAMPLE = '???'
    # Length of each time signal (in signal samples)
    NB_TIME_SAMPLES = '???'
    # DAQ temporal delay between laser excitation and ultrasound recording (in signal samples)
    DAQ_DELAY = '???'
    # Number of samples that are cropped from the beginning of the sinogram (in signal samples)
    CROPPED_SAMPLES = '???'
    # Radius of the detector array (m)
    ARRAY_RADIUS = '???'
    # Angular coverage of the detector array (degrees)
    ARRAY_ANGULAR_COVERAGE = '???'

    ### Field of view
    # Horizontal size of the image (px)
    NB_PIXELS_X = '???'
    # Vertical size of the image (px)
    NB_PIXELS_Z = '???'
    # Horizontal size of the image (m)
    IMA_SIZE_X = '???'
    # Vertical size of the image (m)
    IMA_SIZE_Z = '???'


  # ----------------------------------------------------------------
  # Instantiate the object
  g = geometry.Geometry(
    SCANNER_ID,
    NB_ELEMENTS,
    F_SAMPLE,
    NB_TIME_SAMPLES,
    DAQ_DELAY,
    CROPPED_SAMPLES,
    ARRAY_RADIUS,
    ARRAY_ANGULAR_COVERAGE,
    NB_PIXELS_X,
    NB_PIXELS_Z,
    IMA_SIZE_X,
    IMA_SIZE_Z)

  return g
