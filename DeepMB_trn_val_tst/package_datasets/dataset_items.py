import os
import csv
import torch
import natsort
from medpy.io import load
from torchvision import transforms


class DatasetItems(torch.utils.data.Dataset):

  # ----------------------------------------------------------------
  """
  # **kwargs:
  # Non-empty for the "alternative" training strategy of DeepMB
  # (Training, validation, and testing with 4-1-1 in vivo split)
  # Contains the keys below:
  # ['trn_val_tst_phase'] --> Indicate the current phase of the process, either "trn", "val", or "tst"
  # ['id_val']            --> Indicate which participant belongs to the validation set
  # ['id_tst']            --> Indicate which participant belongs to the testing set
  """
  def __init__(
    self, data_path, targets_folder, divisor_for_input_sinogram_normalization, divisor_for_target_image_normalization,
    samples_limit, regex_print, **kwargs):

    self.path_sinograms = os.path.join(data_path, 'sinograms')
    self.path_targets = os.path.join(data_path, targets_folder)
    self.path_sos = os.path.join(data_path, 'sos_sim_and_rec')
    self.transform_input = transforms.Compose(
      [transforms.ToTensor(), transforms.Normalize(mean=[0], std=[divisor_for_input_sinogram_normalization])])
    self.transform_target = transforms.Compose(
      [transforms.ToTensor(), transforms.Normalize(mean=[0], std=[divisor_for_target_image_normalization])])

    # Count the number of included samples to ensure to not include more than specified
    nb_included_samples = 0

    # Determine if the two files that have been chosen for display examples have been found
    self.found_regex = [False, False]

    # Populate the list containing all the filenames of the dataset
    self.all_data_names = []
    for _, _, f_names in os.walk(self.path_sinograms):
      for f in f_names:
        name_without_extension, _ = os.path.splitext(f)
        append_sample = decide_if_append_sample(name_without_extension, **kwargs)
        # Double condition to determine whether to include a sample or not
        if append_sample and nb_included_samples < samples_limit:
          # Ensure whether both related files (reference image and sos value) exist
          assert_existence_of_target_and_sos_files(name_without_extension, self.path_targets, self.path_sos)
          # Add the current sample to the dataset
          self.all_data_names.append(name_without_extension)
          # Increment the number of included samples so far
          nb_included_samples +=1
          # Verify whether the two files that have been chosen for display examples have been found
          for regex_id in range(2):
            if name_without_extension == regex_print[regex_id]:
              self.found_regex[regex_id] = True

    # Sort the list
    self.all_data_names = natsort.natsorted(self.all_data_names)

  # ----------------------------------------------------------------
  def __len__(self):
    return len(self.all_data_names)


  # ----------------------------------------------------------------
  def __getitem__(self, idx):
    file_name = self.all_data_names[idx]
    input_sinogram, _ = load(os.path.join(self.path_sinograms, file_name + '.nii'))
    target_image, _ = load(os.path.join(self.path_targets, file_name + '.nii'))
    with open(os.path.join(self.path_sos, file_name + '.csv'), 'r') as f:
      csv_reader = csv.reader(f)
      sos = [line for line in csv_reader]
    # Each csv file contains two lines:
    # SoS_ground_truth ("sos_ref"): Optimal SoS value that was determined visually/manually
    # SoS_reconstructed ("sos_rec"): SoS value that was used to reconstruct the reference target image
    sos_ref = float(sos[0][0])
    sos_rec = float(sos[1][0])
    return self.transform_input(input_sinogram), self.transform_target(target_image), file_name, sos_ref, sos_rec


# ----------------------------------------------------------------
def decide_if_append_sample(name_without_extension, **kwargs):

  # If in vivo data is used for training, validation, and testing:
  # Then the sample is only included when the 4-1-1 participants split is respected
  # (see the alternative "in vivo" training strategy described in the DeepMB paper)

  if 'trn_val_tst_phase' in kwargs and 'id_val' in kwargs and 'id_tst' in kwargs:
    trn_val_tst_phase = kwargs['trn_val_tst_phase']
    id_val = kwargs['id_val']
    id_tst = kwargs['id_tst']
    # It is expected that the participant ID is encoded in the 12-nd and 13-rd characters of the filename:
    # For instance: "dmb_orange_p01_biceps_left_longitudinal_01_sosGt_1500_sosRec_1500_700nm"
    id_of_sample = name_without_extension[12:14]
    # Check if the 4-1-1 split is respected
    if \
      (trn_val_tst_phase == 'val' and id_of_sample == id_val) or \
      (trn_val_tst_phase == 'tst' and id_of_sample == id_tst) or \
      (trn_val_tst_phase == 'trn' and not id_of_sample == id_val and not id_of_sample == id_tst):
      append_sample = True
    else:
      append_sample = False

  # If synthetic data is used for training and validation, and in vivo data is used for testing:
  # Then the sample is always included
  # (this is the "standard" training strategy of DeepMB)
  else:
    append_sample = True

  return append_sample


# ----------------------------------------------------------------
def assert_existence_of_target_and_sos_files(name_without_extension, path_targets, path_sos):

  # Ensure whether the reference image file corresponding to the current sinogram exists
  assert os.path.isfile(os.path.join(path_targets, name_without_extension + '.nii')), \
    "File not found: '%s'" % os.path.join(path_targets, name_without_extension + '.nii')

  # Ensure whether the speed of sound information corresponding to the current sinogram exists
  assert os.path.isfile(os.path.join(path_sos, name_without_extension + '.csv')), \
    "File not found: '%s'" % os.path.join(path_sos, name_without_extension + '.csv')
