import os
from .create_folder_if_not_exist import create_folder_if_not_exist


def get_output_folders(save_path, experiment_name, create_missing_folders=False):

  # Root for the current experiment
  experiment_base_path = os.path.join(save_path, experiment_name)

  # Subfolders
  tensorboard_output_dir = os.path.join(save_path, 'tensorboard_logs', experiment_name)
  model_checkpoint_output_dir = os.path.join(experiment_base_path, 'model_checkpoint')
  stored_parameters_output_dir = os.path.join(experiment_base_path, 'stored_parameters')
  print_epoch_dir = os.path.join(experiment_base_path, 'print_epoch_trn_val_tst')
  inference_output_trn_nii_dir = os.path.join(experiment_base_path, 'batch_ifr', 'trn', 'nii')
  inference_output_trn_png_dir = os.path.join(experiment_base_path, 'batch_ifr', 'trn', 'png')
  inference_output_val_nii_dir = os.path.join(experiment_base_path, 'batch_ifr', 'val', 'nii')
  inference_output_val_png_dir = os.path.join(experiment_base_path, 'batch_ifr', 'val', 'png')
  inference_output_tst_nii_dir = os.path.join(experiment_base_path, 'batch_ifr', 'tst', 'nii')
  inference_output_tst_png_dir = os.path.join(experiment_base_path, 'batch_ifr', 'tst', 'png')

  # Create folders if not exist
  if create_missing_folders:
    create_folder_if_not_exist(experiment_base_path)
    create_folder_if_not_exist(tensorboard_output_dir)
    create_folder_if_not_exist(model_checkpoint_output_dir)
    create_folder_if_not_exist(stored_parameters_output_dir)
    create_folder_if_not_exist(print_epoch_dir)
    create_folder_if_not_exist(inference_output_trn_nii_dir)
    create_folder_if_not_exist(inference_output_trn_png_dir)
    create_folder_if_not_exist(inference_output_val_nii_dir)
    create_folder_if_not_exist(inference_output_val_png_dir)
    create_folder_if_not_exist(inference_output_tst_nii_dir)
    create_folder_if_not_exist(inference_output_tst_png_dir)

  return \
    tensorboard_output_dir, \
    model_checkpoint_output_dir, \
    print_epoch_dir, \
    stored_parameters_output_dir, \
    inference_output_trn_nii_dir, \
    inference_output_trn_png_dir, \
    inference_output_val_nii_dir, \
    inference_output_val_png_dir, \
    inference_output_tst_nii_dir, \
    inference_output_tst_png_dir
