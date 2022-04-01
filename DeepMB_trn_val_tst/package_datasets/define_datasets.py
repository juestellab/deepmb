from .get_dataloader import get_dataloader


# ----------------------------------------------------------------
def define_datasets(p, num_workers):

  # DeepMB "standard" case: Training and validations on synthetic data, testing on in vivo data
  if not p.ALTERNATIVE_TRAINING_ON_IN_VIVO_DATA:
    loader_trn, nb_batches_trn, nb_samples_trn, regex_print_trn, \
    loader_val, nb_batches_val, nb_samples_val, regex_print_val, \
    loader_tst, nb_batches_tst, nb_samples_tst, regex_print_tst = define_standard_datasets(p, num_workers)
    flip_trn = False
    flip_val = False
    flip_tst = False
    print('DATASET (Trn/Val): ' + p.DATASET_NAME_TRN_VAL)
    print('DATASET (Tst): ' + p.DATASET_NAME_TST)

  # DeepMB "alternative" case: training, validation, and testing on in vivo data, following a 4/1/1 split
  else:
    loader_trn, nb_batches_trn, nb_samples_trn, regex_print_trn, \
    loader_val, nb_batches_val, nb_samples_val, regex_print_val, \
    loader_tst, nb_batches_tst, nb_samples_tst, regex_print_tst = define_alternative_datasets(p, num_workers)
    flip_trn = False
    flip_val = False
    flip_tst = False
    print('DATASET (in vivo only): ' + p.DATASET_NAME_TST)
    print('Participant ID for validation: ' + p.PARTICIPANT_ID_FOR_VAL)
    print('Participant ID for testing: ' + p.PARTICIPANT_ID_FOR_TST)

  # Console log
  print(
    'Trn:\tsamples: ' + str(nb_samples_trn) +
    ',\t batches: ' + str(nb_batches_trn) +
    ',\t batch size: ' + str(p.BATCH_SIZE_TRN))
  print(
    'Val:\tsamples: ' + str(nb_samples_val) +
    ',\t batches: ' + str(nb_batches_val) +
    ',\t batch size: ' + str(p.BATCH_SIZE_VAL))
  print(
    'Tst:\tsamples: ' + str(nb_samples_tst) +
    ',\t batches: ' + str(nb_batches_tst) +
    ',\t batch size: ' + str(p.BATCH_SIZE_TST))
  print('----------------------------------------------------------------')

  return \
    loader_trn, nb_batches_trn, regex_print_trn, \
    loader_val, nb_batches_val, regex_print_val, \
    loader_tst, nb_batches_tst, regex_print_tst, \
    flip_trn, flip_val, flip_tst


# ----------------------------------------------------------------
# "Standard" training strategy (see DeepMB paper): Training and Validation: synthetic; Testing: in vivo
def define_standard_datasets(p, num_workers):

  loader_trn, nb_batches_trn, nb_samples_trn, regex_print_trn = get_dataloader(
    data_path = p.DATA_PATH,
    dataset_name = p.DATASET_NAME_TRN_VAL,
    trn_val_tst_folder = 'train',
    num_workers = num_workers,
    samples_limit = p.SAMPLES_LIMIT,
    regex_print = [None, None],
    drop_last = False,
    batch_size = p.BATCH_SIZE_TRN,
    shuffle = True,
    target_in_dataset = p.TARGET_IN_DATASET_TRN_VAL,
    divisor_for_input_sinogram_normalization = p.DIVISOR_FOR_INPUT_SINOGRAM_NORMALIZATION,
    divisor_for_target_image_normalization = p.DIVISOR_FOR_TARGET_IMAGE_NORMALIZATION)

  loader_val, nb_batches_val, nb_samples_val, regex_print_val = get_dataloader(
    data_path = p.DATA_PATH,
    dataset_name = p.DATASET_NAME_TRN_VAL,
    trn_val_tst_folder = 'val',
    num_workers = num_workers,
    samples_limit = p.SAMPLES_LIMIT,
    regex_print = p.REGEX_PRINT_VAL,
    drop_last = False,
    batch_size = p.BATCH_SIZE_VAL,
    shuffle = False,
    target_in_dataset = p.TARGET_IN_DATASET_TRN_VAL,
    divisor_for_input_sinogram_normalization = p.DIVISOR_FOR_INPUT_SINOGRAM_NORMALIZATION,
    divisor_for_target_image_normalization = p.DIVISOR_FOR_TARGET_IMAGE_NORMALIZATION)

  loader_tst, nb_batches_tst, nb_samples_tst, regex_print_tst = get_dataloader(
    data_path = p.DATA_PATH,
    dataset_name = p.DATASET_NAME_TST,
    trn_val_tst_folder = 'test',
    num_workers = num_workers,
    samples_limit = p.SAMPLES_LIMIT,
    regex_print = p.REGEX_PRINT_TST,
    drop_last = False,
    batch_size = p.BATCH_SIZE_TST,
    shuffle = False,
    target_in_dataset = p.TARGET_IN_DATASET_TST,
    divisor_for_input_sinogram_normalization = p.DIVISOR_FOR_INPUT_SINOGRAM_NORMALIZATION,
    divisor_for_target_image_normalization = p.DIVISOR_FOR_TARGET_IMAGE_NORMALIZATION)

  return \
    loader_trn, nb_batches_trn, nb_samples_trn, regex_print_trn, \
    loader_val, nb_batches_val, nb_samples_val, regex_print_val, \
    loader_tst, nb_batches_tst, nb_samples_tst, regex_print_tst


# ----------------------------------------------------------------
# "Alternative" training strategy (see DeepMB paper): Training, Validation, and Testing: in vivo, with 4-1-1 split
# The differences from the standard function "define_standard_datasets" are indicated with the mark "[*]"
def define_alternative_datasets(p, num_workers):

  loader_trn, nb_batches_trn, nb_samples_trn, regex_print_trn = get_dataloader(
    data_path = p.DATA_PATH,
    dataset_name = p.DATASET_NAME_TST,  # [*] All data come from the in vivo dataset
    trn_val_tst_folder = 'test',  # [*] This folder contains the in vivo acquisitions
    num_workers = num_workers,
    samples_limit = p.SAMPLES_LIMIT,
    regex_print = [None, None],
    drop_last = False,
    batch_size = p.BATCH_SIZE_TRN,
    shuffle = True,
    target_in_dataset = p.TARGET_IN_DATASET_TRN_VAL,
    divisor_for_input_sinogram_normalization = p.DIVISOR_FOR_INPUT_SINOGRAM_NORMALIZATION,
    divisor_for_target_image_normalization = p.DIVISOR_FOR_TARGET_IMAGE_NORMALIZATION,
    trn_val_tst_phase = 'trn',  # [*] Field to indicate that this dataset is going to serve as the trn/val/tst set
    id_val = p.PARTICIPANT_ID_FOR_VAL,  # [*] ID of the participant that should be included in the validation set
    id_tst = p.PARTICIPANT_ID_FOR_TST)  # [*] ID of the participant that should be included in the test set

  loader_val, nb_batches_val, nb_samples_val, regex_print_val = get_dataloader(
    data_path = p.DATA_PATH,
    dataset_name = p.DATASET_NAME_TST,  # [*] All data come from the in vivo dataset
    trn_val_tst_folder = 'test',  # [*] This folder contains the in vivo acquisitions
    num_workers = num_workers,
    samples_limit = p.SAMPLES_LIMIT,
    regex_print = p.REGEX_PRINT_VAL,
    drop_last = False,
    batch_size = p.BATCH_SIZE_VAL,
    shuffle = False,
    target_in_dataset = p.TARGET_IN_DATASET_TRN_VAL,
    divisor_for_input_sinogram_normalization = p.DIVISOR_FOR_INPUT_SINOGRAM_NORMALIZATION,
    divisor_for_target_image_normalization = p.DIVISOR_FOR_TARGET_IMAGE_NORMALIZATION,
    trn_val_tst_phase = 'val',  # [*] Field to indicate that this dataset is going to serve as the trn/val/tst set
    id_val = p.PARTICIPANT_ID_FOR_VAL,  # [*] ID of the participant that should be included in the validation set
    id_tst = p.PARTICIPANT_ID_FOR_TST)  # [*] ID of the participant that should be included in the test set

  loader_tst, nb_batches_tst, nb_samples_tst, regex_print_tst = get_dataloader(
    data_path = p.DATA_PATH,
    dataset_name = p.DATASET_NAME_TST,  # [*] All data come from the in vivo dataset
    trn_val_tst_folder = 'test',  # [*] This folder contains the in vivo acquisitions
    num_workers = num_workers,
    samples_limit = p.SAMPLES_LIMIT,
    regex_print = p.REGEX_PRINT_TST,
    drop_last = False,
    batch_size = p.BATCH_SIZE_TST,
    shuffle = False,
    target_in_dataset = p.TARGET_IN_DATASET_TST,
    divisor_for_input_sinogram_normalization = p.DIVISOR_FOR_INPUT_SINOGRAM_NORMALIZATION,
    divisor_for_target_image_normalization = p.DIVISOR_FOR_TARGET_IMAGE_NORMALIZATION,
    trn_val_tst_phase = 'tst',  # [*] Field to indicate that this dataset is going to serve as the trn/val/tst set
    id_val = p.PARTICIPANT_ID_FOR_VAL,  # [*] ID of the participant that should be included in the validation set
    id_tst = p.PARTICIPANT_ID_FOR_TST)  # [*] ID of the participant that should be included in the test set

  return \
    loader_trn, nb_batches_trn, nb_samples_trn, regex_print_trn, \
    loader_val, nb_batches_val, nb_samples_val, regex_print_val, \
    loader_tst, nb_batches_tst, nb_samples_tst, regex_print_tst
