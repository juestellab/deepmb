import os
import torch
from .dataset_items import DatasetItems


def get_dataloader(
  data_path, dataset_name, trn_val_tst_folder, num_workers, samples_limit, regex_print, drop_last, batch_size, shuffle,
  target_in_dataset, divisor_for_input_sinogram_normalization, divisor_for_target_image_normalization, **kwargs):

  # Path to the data files
  path_to_data = os.path.join(data_path, dataset_name, trn_val_tst_folder)

  # Dataloader parameters
  params_dataloader = {
    'batch_size': batch_size,
    'shuffle': shuffle,
    'num_workers': num_workers,
    'drop_last': drop_last}

  # Loader
  loader = torch.utils.data.DataLoader(dataset = DatasetItems(
    path_to_data, target_in_dataset, divisor_for_input_sinogram_normalization, divisor_for_target_image_normalization,
    samples_limit, regex_print, **kwargs), **params_dataloader)

  # Count the total number of batches and samples
  nb_batches = len(loader)
  nb_samples = len(loader.dataset)

  # If any of the two files that have been chosen for display examples have not been found...
  # ... the first and/or second file from the available dataset are used instead
  # This is only relevant for the VAL and TST datasets, because the TRN dataset is always re-shuffled ...
  # ... therefore the first and second files are systematically used for display
  for regex_id in range(2):
    if not loader.dataset.found_regex[regex_id] and not regex_print[regex_id] == None:
      regex_print[regex_id] = loader.dataset.all_data_names[min(regex_id, nb_samples)]

  return loader, nb_batches, nb_samples, regex_print
