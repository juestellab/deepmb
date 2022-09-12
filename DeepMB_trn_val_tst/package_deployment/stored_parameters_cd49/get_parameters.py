import parameters


def get_parameters():

  p = parameters.Parameters(

    ### Experiment
    # Unique experiment ID used to log data to tensorboard, print results, and save/load network weights
    EXPERIMENT_NAME = 'cd49_synth_sqrtTargets',

    ### Scanner ID
    # Keyword to define the scanner (see "package_parameters/geometry.py")
    SCANNER_ID = 'Acuity_256e_125d',

    ### Path and data
    # Root folder containing all the different datasets that can be processed
    DATA_PATH = '/media/guillaume/DATA/DeepMB/DATA/dataset_orange_probe/',
    # Folder where the results are be saved (automatically created if it does not already exist)
    SAVE_PATH = '/media/guillaume/DATA/DeepMB/SAVE',
    # Name of the folder containing the training and validation dataset
    DATASET_NAME_TRN_VAL = '210428_dmb_trainval_orange_probe_scaling_0-450_8000train_2000val',
    # Name of the folder containing the testing dataset
    ## DATASET_NAME_TST = '210504_dmb_test_orange_probe',
    DATASET_NAME_TST = '210504_dmb_test_orange_probe',
    # Name of the folder containing the target images for training and validation ("rec_images" or "initial_images")
    TARGET_IN_DATASET_TRN_VAL = 'rec_images',
    # Name of the folder containing the target images for testing ("rec_images" or "initial_images")
    TARGET_IN_DATASET_TST = 'rec_images',
    # If True, "DATASET_NAME_TST" is used for trn, val, and tst, following a 4/1/1 split
    ALTERNATIVE_TRAINING_ON_IN_VIVO_DATA = False,
    # Participant (in vivo data) used for validation (this has no effect if TRAIN_ON_INVIVO_TST_DATA==False)
    PARTICIPANT_ID_FOR_VAL = '00',
    # Participant (in vivo data) used for testing (this has no effect if TRAIN_ON_INVIVO_TST_DATA==False)
    PARTICIPANT_ID_FOR_TST = '00',

    ### Speed of sound
    # Expected SoS values (m/s)
    SOS_RANGE = [1475, 1480, 1485, 1490, 1495, 1500, 1505, 1510, 1515, 1520, 1525],
    # SoS encoding and concatenation (cf "package_tof/concatenate_sos.py" and "package_tof/encode_sos.py")
    SOS_ENCODING = 'ONEHOT_STATIC',
    # Factor to normalize the encoded SoS
    SOS_NORMALIZATION_TO_INPUT_DATA_RANGE = 0.013,

    ### Batch size
    # Number of samples in a training batch
    BATCH_SIZE_TRN = 1,
    # Number of samples in a validation batch
    BATCH_SIZE_VAL = 1,
    # Number of samples in a testing batch
    BATCH_SIZE_TST = 1,

    ### Optimizer and scheduler
    # Optimizer (either 'SGD' or 'Adam')
    OPTIMIZER = 'SGD',
    # Number of epochs
    NB_EPOCH = 300,
    # Learning rate
    LEARNING_RATE = 0.01,
    # Momentum
    MOMENTUM = 0.99,
    # Step size
    SCHEDULER_STEP_SIZE = 1,
    # Gamma
    SCHEDULER_GAMMA = 0.99,

    ### Signal-domain network
    # Network architecture (cf "package_networks/network_DeepMB.py")
    SIG_NET_ARCHI = 'NONE',
    # Enable/disable the bias term
    USE_BIAS_SIG = True,
    # Size of the convolution mask for U-net
    KERNEL_SIZE_UNET_SIG = (3, 3),
    # Padding size for U-net
    PAD_SIZE_UNET_SIG = (1, 1),
    # Number of layers
    NB_LAYERS_UNET_SIG = 5,
    # Number of U-net input channels -- This is not a parameter: by construction, it is 1
    # //! NCH_UNET_SIG = 1
    # Number of U-net filters
    NGF_UNET_SIG = 64,
    # Indicate whether the speed of sound should be provided as a learnable parameter
    PROVIDE_SOS_TO_NET_SIG = False,

    ### Image-domain network
    # Network architecture (cf "package_networks/network_DeepMB.py")
    IMA_NET_ARCHI = 'DAMPING+UNET',
    # Enable/disable the bias term
    USE_BIAS_IMA = True,
    # Size of the convolution mask for U-net
    KERNEL_SIZE_UNET_IMA = (3, 3),
    # Padding size for U-net
    PAD_SIZE_UNET_IMA = (1, 1),
    # Number of U-net layers
    NB_LAYERS_UNET_IMA = 5,
    # Number of U-net input channels -- This is not a parameter: by construction, it is the output of the delay (+sos)
    # //! NCH_UNET_IMA = 256
    # Number of U-net filters
    NGF_UNET_IMA = 64,
    # Gradual reduction of the number of channels via damping (only used if IMA_NET_ARCHI='DAMPING+UNET')
    CH_STEPS_DAMPING = [208, 160, 112],
    # Size of the convolution mask, for the damping network
    KERNEL_SIZE_DAMPING = (3, 3),
    # Size of the convolution padding, for the damping network
    PAD_SIZE_DAMPING = (1, 1),
    # Indicate whether the speed of sound should be provided as a learnable parameter
    PROVIDE_SOS_TO_NET_IMA = True,

    ### Misc
    # Divisor to normalize all input sinograms (e.g., to the range [-1; 1])
    DIVISOR_FOR_INPUT_SINOGRAM_NORMALIZATION = 450,
    # Divisor to normalize all target rec images (e.g., to the range [-1; 1])
    DIVISOR_FOR_TARGET_IMAGE_NORMALIZATION = 450,
    # Data transform applied to normalized targets during trn and val (cf "package_networks/custom_transforms.py")
    TRANSFORM_TRN_VAL_TARGETS = 'SQRT',
    # Data transform applied to the network output during tst (cf "package_networks/custom_transforms.py")
    TRANSFORM_TST_NETWORK_OUTPUT = 'SQUARE',
    # Specify the loss function (cf "package_networks/custom_loss.py")
    LOSS_NICKNAME = 'MSE',
    # Final activation to enforce non-negativity output (cf "package_networks/network_dmb.py")
    OUTPUT_ACTIVATION = 'ABS',
    # If True, the training continues from a previously saved checkpoint (last saved epoch at min val loss)
    RESUME_TRAINING = False,
    # If True, the training starts from a previously saved checkpoint (the architecture may be different)
    USE_PRETRAINING = False,
    # Limit the number of samples in the trn/val/tst datasets (if negative, all samples will be processed)
    SAMPLES_LIMIT = -1,
    # Achieve fully-reproducible results (at the cost a a slightly slower runtime)
    DETERMINISTIC = False,
    # Any number, used to control the randomization for deterministic runs
    RANDOM_SEED = 3.14,
    # Index of the GPU that is used by torch
    GPU_INDEX = 0,

    ### Determine images to be printed out ([displayed in Tensorboard, printed on the hard drive])
    # The training dataset is shuffled anyways, so the first and second IDs area always printed for diversity
    ID_PRINT_TRN = [0, 1],
    # Selected systematic images for the validation dataset (if not found, the last image will be used instead)
    REGEX_PRINT_VAL=[
      '2007_001311_sosGt_1500_sosRec_1500',
      '2007_002293_sosGt_1520_sosRec_1520'],
    # Selected systematic images for the testing dataset (if not found, the last image will be used instead)
    REGEX_PRINT_TST = [
      'dmb_orange_p02_biceps_left_longitudinal_01_sosGt_1500_sosRec_1500_800nm',
      'dmb_orange_p03_ulnar_right_01_sosGt_1490_sosRec_1490_800nm'])

  # Print all attributes in the console for information
  p_attributes = vars(p)
  print('\n'.join("%s: %s" % item for item in p_attributes.items()))
  print('----------------------------------------------------------------')

  # Return an instantiated object from the Parameters class
  return p
